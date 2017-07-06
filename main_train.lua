-- local dbg = require("debugger") -- https://github.com/slembcke/debugger.lua

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image' -- is it necessary?
require 'sys'
-- require 'hdf5'
local utils = require('utils')

local Mesh = require('mesh')
local Hough = require('hough')
local KNN = require('knn')
local hnet = require('hough_net')

local k = 100
local num_of_samples = 1000
local hist_size = 33
local hist_center = 17
local batch_size = 256
local epochs = 50
local train_ratio = 0.95

local batch_size = 64
-- local epochs = 50
local epochs = 30
local train_ratio = 0.85

local base_path = '/home/yanir/Documents/Projects/DeepCloud/'
-- local base_path = '../'

local shape_path = 'data/shapes/'
--local shape_name = {'cube100k','fandisk100k','bunny100k','armadillo100k'}
local shape_name = {'fandisk100k'}
-- local shape_name = {'bunny100k'}
-- local shape_name = {'cube100k','fandisk100k','bunny100k','armadillo100k'}
-- local shape_name = {'cube100k_noise_brown_3e-2','fandisk100k_noise_brown_3e-2','bunny100k_noise_brown_3e-2','armadillo100k_noise_brown_3e-2'}

-- local model_ind = 1
-- -- local out_path = 'data/out/regression_model/'
-- local out_path = 'data/out/regression_model_noise_brown_3e-2/'
-- local learning_rate = 0.001

local model_ind = 2
-- local out_path = 'data/out/classification_model/'
local out_path = 'data/out/classification_model_noise_brown_3e-2/'
local learning_rate = 0.00001

local model_filename = '';
local mean_filename = '';
if #shape_name == 1 then
  model_filename = base_path .. out_path .. shape_name[1] .. '_model.t7'
  mean_filename = base_path .. out_path .. shape_name[1] .. '_mean.t7'
else
  model_filename = base_path .. out_path .. 'model.t7'
  mean_filename = base_path .. out_path .. 'mean.t7'
end

--------------------------------------------------------------------------
---- Iterate over all shapes
local gt_normals = {}
local hough = {}
local pcas = {}
for i,sn in ipairs(shape_name) do
  
  ---- Read shape:
  local xyz_filename = base_path .. shape_path .. sn .. '.xyz'
  local gt_filename = base_path .. shape_path .. sn .. '.normals'

  local v = Mesh.readXYZ(xyz_filename)
  local n = v:size(1)
  
  ---- Read ground truth data:
  local gtn = Mesh.readXYZ(gt_filename)

  --------------------------------------------------------------------------
  ---- Load or compute Hough transform and PCA for each point on the shape:
  local hough_save_name = string.format('%s%s%s_hough_%d_%d.txt', base_path, shape_path, sn, hist_size, num_of_samples)
  local pca_save_name = string.format('%s%s%s_pca_%d_%d.txt', base_path, shape_path, sn, hist_size, num_of_samples)

  local h, p
  if not utils.exists(hough_save_name) or not utils.exists(pca_save_name) then
    h, p = Hough.hough(v, k, num_of_samples, hist_size)

    torch.save(hough_save_name, h, 'ascii')
    torch.save(pca_save_name, p, 'ascii')
    -- local pca_file = hdf5.open(base_path .. out_path .. sn .. '_hough_100.h5', 'w')
    -- pca_file:write('hough', h)
    -- pca_file:close()
    -- local hough_file = hdf5.open(base_path .. out_path .. sn .. '_pca_100.h5', 'w')
    -- hough_file:write('pcas', p)
    -- hough_file:close()
  else
    sys.tic()
    
    h = torch.load(hough_save_name, 'ascii')
    p = torch.load(pca_save_name, 'ascii')
    
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

-- normalize each sample based on its maximum value:
hmax = hough:max(2):expandAs(hough)
hough:cdiv(hmax)

---- normalize hough transform:
--hough:div(num_of_samples)

-- Change input size: 1 input layer (channel), hist_size * hist_size image:
hough = hough:reshape(hough:size(1), 1, hist_size, hist_size)



-- Transform 3D ground truth normals to deep net 2D normals using PCAs:
if model_ind == 2 then
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

local model = nil
if model_ind == 2 then
    model = hnet.getModel2()
else
    model = hnet.getModel()
end

print('Initialized model in ' .. sys.toc() .. ' seconds.')

------------------------------------------------------------------------
---- Train deep net:
sys.tic()
model = hnet.train(hough_train, gt_train, model, batch_size, epochs, learning_rate, model_ind)

print('Trained model in ' .. sys.toc() .. ' seconds.')
sys.tic()

-- save current net:
torch.save(model_filename, model)

print('Saved model in ' .. sys.toc() .. ' seconds.')

