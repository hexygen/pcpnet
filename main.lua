require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image' -- is it necessary?
require 'sys'
local utils = require('utils')

local Mesh = require('mesh')
local Hough = require('hough')
local KNN = require('knn')
local hnet = require('hough_net')

local k = 100
local num_of_samples = 1000
local hist_size = 33
local batch_size = 1000

local base_path = '/home/yanir/Documents/Projects/DeepCloud/'
local shape_path = 'data/shapes/'
local shape_name = '151A_100k_0005'
local out_path = 'data/out/'
local model_path = 'data/model_1s/'

local xyz_filename = base_path .. shape_path .. shape_name .. '.xyz'
local output_filename = base_path .. out_path .. shape_name .. '_normals.xyz'

--------------------------------------------------------------------------
---- Read shape:
local v = Mesh.readXYZ(xyz_filename)
local n = v:size(1)

--------------------------------------------------------------------------
---- Load or compute Hough transform and PCA for each point on the shape:
local hough_save_name = base_path .. out_path .. shape_name .. '_hough_100.txt'
local pca_save_name = base_path .. out_path .. shape_name .. '_pca_100.txt'

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



------------------------------------------------------------------------
---- Load models of deep net:
sys.tic()

local model_name = base_path .. model_path .. 'net.t7'
local mean_name = base_path .. model_path .. 'mean.t7'
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
local normals = hnet.evaluate(hough, model, batch_size)

-- Transform 2D output of deep net to 3D normals:
Hough.postprocess_normals(normals, pcas)

-- Write output file:
Mesh.writeXYZ(output_filename, v, normals)
