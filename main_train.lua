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
local batch_size = 64
local epochs = 50
local train_ratio = 0.85

local base_path = '/home/yanir/Documents/Projects/DeepCloud/'
-- local base_path = '../'

local shape_path = 'data/shapes/'
local shape_name = 'cube100k'
local model_path = 'data/model_1s/'

local model_ind = 1
local out_path = 'data/out/'
local learning_rate = 0.001

-- local model_ind = 2
-- local out_path = 'data/out/model2/'
-- local learning_rate = 0.00001

local xyz_filename = base_path .. shape_path .. shape_name .. '.xyz'
local gt_filename = base_path .. shape_path .. shape_name .. '.normals'
local output_filename = base_path .. out_path .. shape_name .. '_normals.xyz'
local model_filename = base_path .. out_path .. shape_name .. '_model.t7'
local mean_filename = base_path .. out_path .. shape_name .. '_mean.t7'

--------------------------------------------------------------------------
---- Read shape:
local v = Mesh.readXYZ(xyz_filename)
local n = v:size(1)
---- Read ground truth data:
local gt_normals = Mesh.readXYZ(gt_filename)

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
---- Preprocess data - split into train and test set:
sys.tic()

-- Change input size: 1 input layer (channel), hist_size * hist_size image:
hough = hough:reshape(hough:size(1), 1, hist_size, hist_size)
-- normalize hough transform:
hough:div(num_of_samples)

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
--local model_name = base_path ..out_path .. 'model.t7'
--local mean_name = base_path .. model_path .. 'mean.t7'
--local mean = torch.load(mean_name):float()
--local model = torch.load(model_name)

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

