-- local dbg = require("debugger") -- https://github.com/slembcke/debugger.lua

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image' -- is it necessary?
require 'sys'
require 'hdf5'
require 'os'
local utils = require('utils')

local Mesh = require('mesh')
local Hough = require('hough')
local KNN = require('knn')
local hnet = require('hough_net')
local model_list = require('model_list')

local k = 100
local num_of_samples = 1000
local hist_size = 33
local batch_size = 256

-- local base_path = '/home/yanir/Documents/Projects/DeepCloud/'
local base_path = '../'

local shape_path = 'data/shapes/'

local shapes = {}
shapes['t'] = '151A_100k_0005'
shapes['c'] = 'cube100k'
shapes['f'] = 'fandisk100k'
shapes['b'] = 'bunny100k'
shapes['a'] = 'armadillo100k'
shapes['d'] = 'dragon100k'
shapes['h'] = 'happy100k'

for k,v in pairs(shapes) do
  shapes[k .. 'n'] = shapes[k] .. '_noise_brown_3e-2'
end

------------------------------------------------------------------
---- For day to day testing change only the following 2-3 lines:
------------------------------------------------------------------
-- Shapes to run - change this to evaluate specific shapes:
local run_shapes = {'c','f','b','a','d','h','cn','fn','bn','an','dn','hn'}
-- Model to run - change this to evaluate on different models:
local run_models = {model_list['re3'], model_list['re4']}
-- Optional: Prediction method with a given model (leave empty if there is only one prediction method)
local pred_method = '';
------------------------------------------------------------------

for x, model in ipairs(run_models) do

  local model_path = 'data/out/' .. model['id'] .. '/'
  local out_path = model_path
  if string.len(pred_method) > 0 then
    out_path = out_path .. pred_method .. '/'
  end

  local model_filename = base_path .. model_path .. 'model.t7'
  local mean_filename = base_path .. model_path .. 'mean.t7'


  for i,sid in ipairs(run_shapes) do
    local sn = shapes[sid]

    local xyz_filename = base_path .. shape_path .. sn .. '.xyz'
    local gt_filename = base_path .. shape_path .. sn .. '.normals'

    local output_filename = base_path .. out_path .. sn .. '_normals.xyz'

    --------------------------------------------------------------------------
    ---- Read shape:
    local v = Mesh.readXYZ(xyz_filename)
    local n = v:size(1)
    ---- Read ground truth data:
    local gt = Mesh.readXYZ(gt_filename)

    --------------------------------------------------------------------------
    ---- Load or compute Hough transform and PCA for each point on the shape:
    local hough_save_name = string.format('%s%s%s_hough_%d_%d.h5', base_path, shape_path, sn, hist_size, num_of_samples)
    local pca_save_name = string.format('%s%s%s_pca_%d_%d.h5', base_path, shape_path, sn, hist_size, num_of_samples)

    local hough, pcas
    if not utils.exists(hough_save_name) or not utils.exists(pca_save_name) then
      hough, pcas = Hough.hough(v, k, num_of_samples, hist_size)

      -- torch.save(hough_save_name, hough, 'ascii')
      -- torch.save(pca_save_name, pcas, 'ascii')

      local h5file = hdf5.open(hough_save_name, 'w')
      h5file:write('hough', hough)
      h5file:close()
      local h5file = hdf5.open(pca_save_name, 'w')
      h5file:write('pcas', pcas)
      h5file:close()

    else
      sys.tic()

      -- hough = torch.load(hough_save_name, 'ascii')
      -- pcas = torch.load(pca_save_name, 'ascii')

      local h5file = hdf5.open(hough_save_name, 'r')
      hough = h5file:read('hough'):all()
      h5file:close()
      local h5file = hdf5.open(pca_save_name, 'r')
      pcas = h5file:read('pcas'):all()
      h5file:close()

      print('Loaded Hough transform and PCA from file in ' .. sys.toc() .. ' seconds.')
    end

    -- save hough and pcas in hdf5 format as well (so they can be loaded into matlab)
  --  local hough_save_name_h5 = utils.splitext(hough_save_name) .. '.h5'
  --  local pca_save_name_h5 = utils.splitext(pca_save_name) .. '.h5'
  --  if not utils.exists(hough_save_name) or not utils.exists(pca_save_name_h5) then
  --    local hough_file = hdf5.open(hough_save_name_h5, 'w')
  --    hough_file:write('hough', hough)
  --    hough_file:close()

  --    local pca_file = hdf5.open(pca_save_name_h5, 'w')
  --    pca_file:write('pcas', pcas)
  --    pca_file:close()
  --  end

    local normals = {}
    if model['method'] == 'pca' then
      -- No need to call the network - return zero and do reverse PCA:
      normals = torch.zeros(n,2):float()
    else
      ------------------------------------------------------------------------
      ---- Load models of deep net:
      sys.tic()

      local mean = torch.load(mean_filename):float()
      local model = torch.load(model_filename)

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
      normals = hnet.evaluate(hough, model, batch_size, model['method'])
    end

    -- Transform 2D output of deep net to 3D normals:
    if model['method'] == 'cl' then
      -- local hough_file = hdf5.open(base_path .. out_path .. sn .. '_hough_opt.h5', 'w')
      -- hough_file:write('hough_opt', normals)
      -- hough_file:close()
      normals = Hough.postprocess_normals2(normals, pcas, hist_size, pred_method)
    else
      normals = Hough.postprocess_normals(normals, pcas)
    end

    -- Write output file:
    Mesh.writeXYZ(output_filename, v, normals)

  end  -- for each shape
end -- for each model