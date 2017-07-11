-- local dbg = require("debugger") -- https://github.com/slembcke/debugger.lua

require 'cutorch'
require 'nngraph'
require 'sys'
require 'optim'

local M={};

--[[

Network printed structure:

nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
  (1): cudnn.SpatialConvolution(1 -> 50, 3x3)
  (2): cudnn.ReLU
  (3): nn.SpatialBatchNormalization (4D) (50)
  (4): cudnn.SpatialConvolution(50 -> 50, 3x3)
  (5): cudnn.ReLU
  (6): nn.SpatialBatchNormalization (4D) (50)
  (7): cudnn.SpatialMaxPooling(2x2, 2,2)
  (8): cudnn.SpatialConvolution(50 -> 96, 3x3)
  (9): cudnn.ReLU
  (10): cudnn.SpatialMaxPooling(2x2, 2,2)
  (11): nn.View(-1)
  (12): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> output]
    (1): nn.Dropout(0.500000)
    (2): nn.Linear(3456 -> 2048)
    (3): cudnn.ReLU
    (4): nn.Dropout(0.500000)
    (5): nn.Linear(2048 -> 1024)
    (6): cudnn.ReLU
    (7): nn.Dropout(0.500000)
    (8): nn.Linear(1024 -> 512)
    (9): cudnn.ReLU
    (10): nn.Linear(512 -> 2)
  }
}


]]

function M.getModel()

  local model = nn.Sequential();
  model:add(cudnn.SpatialConvolution(1, 50, 3, 3)) -- first parameter is the number of multi-scale layers.
  model:add(cudnn.ReLU())
  model:add(nn.SpatialBatchNormalization(50)) -- IS this correct? Is the (4D) just an indication or a parameter?
  model:add(cudnn.SpatialConvolution(50, 50, 3, 3))
  model:add(cudnn.ReLU())
  model:add(nn.SpatialBatchNormalization(50))
  model:add(cudnn.SpatialMaxPooling(2, 2))
  model:add(cudnn.SpatialConvolution(50, 96, 3, 3))
  model:add(cudnn.ReLU())
  model:add(cudnn.SpatialMaxPooling(2,2))
  model:add(nn.View(-1):setNumInputDims(3))

  local seq2 = nn.Sequential()

  seq2:add(nn.Dropout())
  seq2:add(nn.Linear(3456, 2048))
  seq2:add(cudnn.ReLU())
  seq2:add(nn.Dropout())
  seq2:add(nn.Linear(2048, 1024))
  seq2:add(cudnn.ReLU())
  seq2:add(nn.Dropout())
  seq2:add(nn.Linear(1024, 512))
  seq2:add(cudnn.ReLU())
  seq2:add(nn.Linear(512, 2))


  -- Consider adding Tanh to restrict values to [-1, 1]?
  -- Or add a normalization layer? nn.Normalize?

  model:add(seq2)

  return model
end

function M.getModel2()

  -- -- autoencoder architecture without skip connections
  -- local model = nn.Sequential();
  -- model:add(cudnn.SpatialConvolution(1, 32, 3, 3,2,2,1,1)) -- 32 x 17 x 17 output
  -- model:add(nn.SpatialBatchNormalization(32))
  -- model:add(nn.LeakyReLU(0.2))
  -- model:add(cudnn.SpatialConvolution(32, 64, 3, 3,2,2,1,1)) -- 64 x 9 x 9 output
  -- model:add(nn.SpatialBatchNormalization(64))
  -- model:add(nn.LeakyReLU(0.2))
  -- model:add(cudnn.SpatialConvolution(64, 128, 3, 3,2,2,1,1)) -- 128 x 5 x 5 output
  -- model:add(nn.SpatialBatchNormalization(128))
  -- model:add(nn.LeakyReLU(0.2))
  -- model:add(cudnn.SpatialConvolution(128, 256, 3, 3,2,2,1,1)) -- 256 x 3 x 3 output
  -- model:add(nn.SpatialBatchNormalization(256))
  -- model:add(nn.LeakyReLU(0.2))
  -- model:add(cudnn.SpatialConvolution(256, 256, 3, 3,1,1,1,1)) -- 256 x 3 x 3 output
  -- model:add(nn.SpatialBatchNormalization(256))
  -- model:add(nn.LeakyReLU(0.2))
  -- model:add(nn.SpatialUpSamplingNearest(2)) -- 128 x 6 x 6 output
  -- model:add(cudnn.SpatialConvolution(256, 128, 4, 4,1,1,1,1)) -- 128 x 5 x 5 output
  -- model:add(nn.SpatialBatchNormalization(128))
  -- model:add(nn.LeakyReLU(0.2))
  -- model:add(nn.SpatialUpSamplingNearest(2)) -- 64 x 10 x 10 output
  -- model:add(cudnn.SpatialConvolution(128, 64, 4, 4,1,1,1,1)) -- 64 x 9 x 9 output
  -- model:add(nn.SpatialBatchNormalization(64))
  -- model:add(nn.LeakyReLU(0.2))
  -- model:add(nn.SpatialUpSamplingNearest(2)) -- 64 x 18 x 18 output
  -- model:add(cudnn.SpatialConvolution(64, 32, 4, 4,1,1,1,1)) -- 32 x 17 x 17 output
  -- model:add(nn.SpatialBatchNormalization(32))
  -- model:add(nn.LeakyReLU(0.2))
  -- model:add(nn.SpatialUpSamplingNearest(2)) -- 64 x 34 x 34 output
  -- model:add(cudnn.SpatialConvolution(32, 16, 4, 4,1,1,1,1)) -- 16 x 33 x 33 output
  -- model:add(nn.SpatialBatchNormalization(16))
  -- model:add(nn.LeakyReLU(0.2))
  -- model:add(cudnn.SpatialConvolution(16, 1, 3, 3,1,1,1,1)) -- 1 x 33 x 33 output
  -- model:add(nn.Sigmoid())

  -- autoencoder architecture with skip connections
  local e0 = - nn.Identity() -- 1 x 33 x 33 output
  local e1 = e0 - cudnn.SpatialConvolution(1, 32, 3, 3,2,2,1,1) -- 32 x data17 x 17 output
  local e2 = e1 - nn.LeakyReLU(0.2, true) - cudnn.SpatialConvolution(32, 64, 3, 3,2,2,1,1) - cudnn.SpatialBatchNormalization(64) -- 64 x 9 x 9 output
  local e3 = e2 - nn.LeakyReLU(0.2, true) - cudnn.SpatialConvolution(64, 128, 3, 3,2,2,1,1) - cudnn.SpatialBatchNormalization(128) -- 128 x 5 x 5 output
  local e4 = e3 - nn.LeakyReLU(0.2, true) - cudnn.SpatialConvolution(128, 256, 3, 3,2,2,1,1) - cudnn.SpatialBatchNormalization(256) -- 256 x 3 x 3 output
  local e5 = e4 - nn.LeakyReLU(0.2, true) - cudnn.SpatialConvolution(256, 256, 3, 3,1,1,1,1) - cudnn.SpatialBatchNormalization(256) -- 256 x 3 x 3 output

  local d1_ = e5 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(256, 128, 4, 4,1,1,1,1) - cudnn.SpatialBatchNormalization(128) -- 128 x 5 x 5 output
  local d1 = {d1_,e3} - nn.JoinTable(2)
  local d2_ = d1 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(256, 64, 4, 4,1,1,1,1) - cudnn.SpatialBatchNormalization(64) -- 64 x 9 x 9 output
  local d2 = {d2_,e2} - nn.JoinTable(2)
  local d3_ = d2 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(128, 32, 4, 4,1,1,1,1) - cudnn.SpatialBatchNormalization(32) -- 32 x 17 x 17 output
  local d3 = {d3_,e1} - nn.JoinTable(2)
  local d4_ = d3 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(64, 16, 4, 4,1,1,1,1) - cudnn.SpatialBatchNormalization(16) -- 16 x 33 x 33 output
  local d4 = {d4_,e0} - nn.JoinTable(2)
  local d5 = d4 - nn.ReLU(true) - cudnn.SpatialConvolution(17, 1, 3, 3,1,1,1,1) -- 1 x 33 x 33 output

  local o1 = d5 - nn.Sigmoid()

  local model = nn.gModule({e0},{o1})

  -- custom conv and batchnorm layer initialization
  M.initModelparams(model)

  return model
end

function M.initLayerparams(layer)
    local layertype = torch.type(layer)
    if layertype:find('Convolution') then
        -- m.weight:normal(0.0, 0.02)
        -- m.bias:fill(0)

        -- initialize weights to a normal distribution with variance 2 / (number of neuron inputs) and bias to zero
        layer.weight:normal(0.0,torch.sqrt(2.0 / (layer.nInputPlane*layer.kW*layer.kH)))
        layer.bias:zero()
    elseif layertype:find('BatchNormalization') then
        -- initialize weights to a normal distribution with variance 0.02 and bias to zero
        if layer.weight then layer.weight:normal(1.0, 0.02) end
        if layer.bias then layer.bias:zero() end
    end
end

function M.initModelparams(model)
    model:apply(M.initLayerparams)
end

-- Trains the given model using the samples and the ground truth data.
function M.train(samples, gt, model, batch_size, epochs, learning_rate, method)

  -- Just in case:
  samples = samples:float()
  gt = gt:float()

  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  model:training()
  model:cuda()

  -- Get model parameters:
  local params, gradParams = model:getParameters()
  -- Set optimization parameters:
  local optimState = {learningRate = learning_rate}

  ---- Define the mean squared error crieterion:
  local criterion = nn.MSECriterion():cuda()
  criterion.sizeAverage = false


  local n = samples:size(1)
  local bn = math.floor(n / batch_size) -- number of batches
  local nn = bn * batch_size       -- number of samples that fit in each epoch

  -- length of sizes vector:
  local s_size = samples:size()
  local sizes_len = s_size:size()
  local size_batches = torch.LongStorage(sizes_len + 1)
  size_batches[1] = bn
  size_batches[2] = batch_size
  for i=3,sizes_len+1 do
    size_batches[i] = s_size[i-1]
  end


  -- Initialize variables so the space will only be allocated once:
  local shuffled = torch.FloatTensor()
  local gt_shuffled = torch.FloatTensor()


  for epoch = 1, epochs do

    -- shuffle at each epoch
    local shuffle_ind = torch.randperm(n):long()
    -- Getting only the first nn samples, discarding the residual:
    shuffled:index(samples, 1, shuffle_ind[{{1, nn}}])
    gt_shuffled:index(gt, 1, shuffle_ind[{{1, nn}}])

    -- split into batches using view:
    local batches = shuffled:view(size_batches)
    -- output is a 2D normal so the dimension is 2:
    local gt_batches
    if method == 'cl' then
      gt_batches = gt_shuffled:view(size_batches)
    else
      gt_batches = gt_shuffled:view(bn, batch_size, 2)
    end

    for b = 1, bn do

      -- local function we give to optim
      -- it takes current weights as input, and outputs the loss
      -- and the gradient of the loss with respect to the weights
      -- gradParams is calculated implicitly by calling 'backward',
      -- because the model's weight and bias gradient tensors
      -- are simply views onto gradParams
      function feval(params)
        -- just in case:
        collectgarbage()

        gradParams:zero()

        local inputs = batches[b]:cuda()
        local gts = gt_batches[b]:cuda()

        local outputs = model:forward(inputs)
        local loss = criterion:forward(outputs, gts)
        local dloss_doutputs = criterion:backward(outputs, gts)
        model:backward(inputs, dloss_doutputs)

        out_norm = torch.sum(torch.norm(outputs, 2, 2))
        if (math.fmod(b, 5) == 0) then
          print('Loss = ' .. loss .. ', Norm = ' .. out_norm)
        end
        
        return loss, gradParams
      end

      --- TODO: Check parameters for params optimState
      optim.sgd(feval, params, optimState)

      if (math.fmod(b, 50) == 0) then
        print('Training: epoch '.. epoch .. ' batch ' .. b .. ' points = ' .. (b*batch_size))
      end

    end

    if (math.fmod(epoch, 3) == 0) then
      optimState.learningRate = optimState.learningRate * 0.8
    end
    

  end

  return model

end



function M.evaluate(samples, model, batch_size, method)

  local n = samples:size(1)

  -- Set module to 'Evaluate' mode in contrast with training mode:
  model:evaluate()

  local ind_start = 1
  local ind_end = batch_size
  
  local outputs = nil
  if method == 'cl' then
    outputs = torch.FloatTensor(n, samples:size(3) * samples:size(4))
  else
    outputs = torch.FloatTensor(n, 2)
  end

  while (ind_start < n) do

    if (ind_end > n) then
      ind_end = n
    end

    -- forward (make sure input is cuda)
    local chunk = samples[{{ind_start, ind_end}}]
    local out_chunk = model:forward(chunk:cuda())

    outputs[{{ind_start, ind_end}, {}}] = out_chunk:float()

    print('Forward step batch ' .. ind_end .. ' in ' .. sys.toc() .. ' seconds.')

    ind_start = ind_start + batch_size
    ind_end = ind_end + batch_size

  end

  return outputs
end


return M;
