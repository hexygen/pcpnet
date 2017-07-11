-- local dbg = require("debugger") -- https://github.com/slembcke/debugger.lua

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image' -- is it necessary?
require 'sys'
require 'hdf5'
require 'lfs'
local utils = require('utils')

local Mesh = require('mesh')
local Hough = require('hough')
local KNN = require('knn')
local hnet = require('hough_net')
local model_list = require('model_list')


-- local base_path = '/home/yanir/Documents/Projects/DeepCloud/'
local base_path = '../'

local shape_path = 'data/shapes/'

-- \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ --
-- Change model id to train a different network:
local model_ids = {"cl1", "cln1"}
-- /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\  --

-- Allow training several networks one after another on a lunch break:
for i,model_id in ipairs(model_ids) do
  
  local model = model_list[model_id]
  local params = model["parameters"]

  -- Parameters are taken from the model list. 
  -- To change training parameters add a new model to the list.
  local k = params["k"]
  local num_of_samples = params["num_of_samples"]
  local hist_size = params["hist_size"]
  local hist_center = params["hist_center"]
  local batch_size = params["batch_size"]
  local epochs = params["epochs"]
  local train_ratio = params["train_ratio"]
  local learning_rate = params["learning_rate"]

  local shapes = model["shapes"]

  local out_path = base_path .. 'data/out/' .. model_id .. '/'
  if not lfs.attributes(out_path) then
    local success = lfs.mkdir(out_path)
    if not success then
      print('Error creating folder for model ' .. model_id)
      break
    end
  end
  

  local model_filename = out_path .. 'model.t7'
  local mean_filename = out_path .. 'mean.t7'

  --------------------------------------------------------------------------
  ---- Iterate over all shapes
  local gt_normals = {}
  local hough = {}
  local pcas = {}
  for i,sn in ipairs(shapes) do
    
    ---- Read shape:
    local xyz_filename = base_path .. shape_path .. sn .. '.xyz'
    local gt_filename = base_path .. shape_path .. sn .. '.normals'

    local v = Mesh.readXYZ(xyz_filename)
    local n = v:size(1)
    
    ---- Read ground truth data:
    local gtn = Mesh.readXYZ(gt_filename)

    --------------------------------------------------------------------------
    ---- Load or compute Hough transform and PCA for each point on the shape:
    local hough_save_name = string.format('%s%s%s_hough_%d_%d.h5', base_path, shape_path, sn, hist_size, num_of_samples)
    local pca_save_name = string.format('%s%s%s_pca_%d_%d.h5', base_path, shape_path, sn, hist_size, num_of_samples)

    local h, p
    if not utils.exists(hough_save_name) or not utils.exists(pca_save_name) then
      h, p = Hough.hough(v, k, num_of_samples, hist_size)

      -- torch.save(hough_save_name, h, 'ascii')
      -- torch.save(pca_save_name, p, 'ascii')
      
      local h5file = hdf5.open(hough_save_name, 'w')
      h5file:write('hough', h)
      h5file:close()
      local h5file = hdf5.open(pca_save_name, 'w')
      h5file:write('pcas', p)
      h5file:close()
    else
      sys.tic()
      
      -- h = torch.load(hough_save_name, 'ascii')
      -- p = torch.load(pca_save_name, 'ascii')

      local h5file = hdf5.open(hough_save_name, 'r')
      h = h5file:read('hough'):all()
      h5file:close()
      local h5file = hdf5.open(pca_save_name, 'r')
      p = h5file:read('pcas'):all()
      h5file:close()
      
      print('Loaded Hough transform and PCA from file in ' .. sys.toc() .. ' seconds.')
    end
    
    ---- Append gt, hough and pca of current shape to input list
    if i==1 then
      gt_normals = gtn
      hough = h
      pcas = p
    else
      gt_normals = torch.cat(gt_normals,gtn,1)
      hough = torch.cat(hough,h,1)
      pcas = torch.cat(pcas,p,1)
    end

  end

  local n = hough:size(1)

  ------------------------------------------------------------------------
  ---- Preprocess data - split into train and test set:
  sys.tic()

  -- Find points next to edges (cube only):
  --vm = v:abs():median()
  --is_edgy = vm:gt(0.75):cmul(vm:lt(0.99))
  --ind_edgy = torch.nonzero(is_edgy):select(2, 1)


  -- Exclude flat points from training set:
  hough_center = hough:reshape(hough:size(1), 1, hist_size, hist_size)[{{},{}, hist_center, hist_center}]
  -- If all samples are in the center cell, this is a flat area:
  is_nonflat = hough_center:ne(num_of_samples)
  ind_nonflat = torch.nonzero(is_nonflat):select(2, 1)

  hough = hough:index(1, ind_nonflat)
  gt_normals = gt_normals:index(1, ind_nonflat)
  pcas = pcas:index(1, ind_nonflat)

  n = ind_nonflat:size(1)

  -- a nil value will be considered as false here:
  if params["use_num_of_samples"] then
    -- normalize hough transform:
    hough:div(num_of_samples)
  else
    -- normalize each sample based on its maximum value:
    hmax = hough:max(2):expandAs(hough)
    hough:cdiv(hmax)
  end

  -- Change input size: 1 input layer (channel), hist_size * hist_size image:
  hough = hough:reshape(hough:size(1), 1, hist_size, hist_size)

  -- Transform 3D ground truth normals to deep net 2D normals using PCAs:
  if model['method'] == 'cl' then
      gt = Hough.preprocess_normals2(gt_normals, pcas, hist_size)
  else
      gt = Hough.preprocess_normals(gt_normals, pcas)
  end
  print('Rotated normals in ' .. sys.toc() .. ' seconds.')


  local shuffle = torch.randperm(n):long()

  local train_size = math.ceil(n * train_ratio)

  local hough_train = hough:index(1, shuffle[{{1, train_size}}])
  local gt_train = gt:index(1, shuffle[{{1, train_size}}])
  local hough_test = hough:index(1, shuffle[{{train_size+1, n}}])
  local gt_test = gt:index(1, shuffle[{{train_size+1, n}}])

  ---- Normalize data:
  local mean = torch.mean(hough_train, 1)
  -- Substract mean
  hough:add(-1, mean:expandAs(hough))
  -- save current net:
  torch.save(mean_filename, mean)

  -- Should we divide by std? Original code did not!
  print('Preprocessed data in ' .. sys.toc() .. ' seconds.')


  ------------------------------------------------------------------------
  ---- Initialize model of deep net:
  sys.tic()

  local net = nil
  if model['method'] == 'cl' then
      net = hnet.getModel2()
  else
      net = hnet.getModel()
  end

  print('Initialized model in ' .. sys.toc() .. ' seconds.')

  ------------------------------------------------------------------------
  ---- Train deep net:
  sys.tic()
  net = hnet.train(hough_train, gt_train, net, batch_size, epochs, learning_rate, model['method'])

  print('Trained model in ' .. sys.toc() .. ' seconds.')
  sys.tic()

  -- save current net:
  torch.save(model_filename, net)

  print('Saved model in ' .. sys.toc() .. ' seconds.')

end