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
local batch_size = 256

-- local base_path = '/home/yanir/Documents/Projects/DeepCloud/'
local base_path = '../'

local shape_path = 'data/shapes/'
--local shape_name = {'151A_100k_0005'}
local shape_name = {'cube100k','fandisk100k'}

-- -- pca normals
-- local model_ind = 0
-- local out_path = 'data/out/pca_only/'

-- local model_ind = 1
-- local out_path = 'data/out/regression_model/'

local model_ind = 2
local out_path = 'data/out/classification_model/'

for i,sn in ipairs(shape_name) do
  local xyz_filename = base_path .. shape_path .. sn .. '.xyz'
  local gt_filename = base_path .. shape_path .. sn .. '.normals'

  -- -- boulch model
  -- local model_name = base_path .. 'data/model_1s/net.t7'
  -- local mean_name = base_path .. 'data/model_1s/mean.t7'
  -- local output_filename = base_path .. out_path .. sn .. '_normals_boulch.xyz'

  -- our model
  local model_name = base_path .. out_path .. 'model.t7'
  local mean_name = base_path .. out_path .. 'mean.t7'
  local output_filename = base_path .. out_path .. sn .. '_normals.xyz'
  -- local output_filename = base_path .. out_path .. sn .. '_normals_avg.xyz'

  --------------------------------------------------------------------------
  ---- Read shape:
  local v = Mesh.readXYZ(xyz_filename)
  local n = v:size(1)
  ---- Read ground truth data:
  local gt = Mesh.readXYZ(gt_filename)

  --------------------------------------------------------------------------
  ---- Load or compute Hough transform and PCA for each point on the shape:
  local hough_save_name = base_path .. out_path .. sn .. '_hough_100.txt'
  local pca_save_name = base_path .. out_path .. sn .. '_pca_100.txt'

  local hough, pcas
  if not utils.exists(hough_save_name) then
    hough, pcas = Hough.hough(v, k, num_of_samples, hist_size)

    torch.save(hough_save_name, hough, 'ascii')
    torch.save(pca_save_name, pcas, 'ascii')
  else
    sys.tic()
    
    hough = torch.load(hough_save_name, 'ascii')
    pcas = torch.load(pca_save_name, 'ascii')
    
    print('Loaded Hough transform and PCA from file in ' .. sys.toc() .. ' seconds.')
  end

  local normals = {}
  if model_ind == 0 then
      normals = torch.zeros(n,2):float()
  else
      ------------------------------------------------------------------------
      ---- Load models of deep net:
      sys.tic()

      local mean = torch.load(mean_name):float()
      local model = torch.load(model_name)

      print('Loaded model in ' .. sys.toc() .. ' seconds.')

      ------------------------------------------------------------------------
      ---- Substract the mean from the hough transform (not sure why...):
      sys.tic()

      -- Resize to necessary size and copy to float:
      hough = hough:resize(n, 1, hist_size, hist_size):float()
      -- normalize hough transform:
      hough:div(num_of_samples)

      -- substract mean
      local mean_exp = mean:resize(1, 1, hist_size, hist_size):expandAs(hough)
      hough:add(-1, mean_exp)

      mean = nil
      mean_exp = nil

      collectgarbage()

      print('Substracted mean in ' .. sys.toc() .. ' seconds.')

      ------------------------------------------------------------------------
      ---- Evaluate deep net:

      normals = hnet.evaluate(hough, model, batch_size, model_ind)
  end

  -- Compute normals with pca, i.e. output zeros before post processing:
  --local normals = torch.FloatTensor(n, 2)

  -- Transform 2D output of deep net to 3D normals:
  if model_ind == 2 then
      -- local hough_file = hdf5.open(base_path .. out_path .. sn .. '_hough_opt.h5', 'w')
      -- hough_file:write('hough_opt', normals)
      -- hough_file:close()
      normals = Hough.postprocess_normals2(normals, pcas, hist_size)
  else
      normals = Hough.postprocess_normals(normals, pcas)
  end

  -- Write output file:
  Mesh.writeXYZ(output_filename, v, normals)

end
