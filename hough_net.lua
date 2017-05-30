require 'cutorch'
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

  --seq2:add(nn.Dropout())
  seq2:add(nn.Linear(3456, 2048))
  seq2:add(cudnn.ReLU())
  --seq2:add(nn.Dropout())
  seq2:add(nn.Linear(2048, 1024))
  seq2:add(cudnn.ReLU())
  --seq2:add(nn.Dropout())
  seq2:add(nn.Linear(1024, 512))
  seq2:add(cudnn.ReLU())
  seq2:add(nn.Linear(512, 2))


  -- Consider adding Tanh to restrict values to [-1, 1]?
  -- Or add a normalization layer? nn.Normalize?

  model:add(seq2)

  return model
end

-- Trains the given model using the samples and the ground truth data.
function M.train(samples, gt, model, batch_size, epochs)

  -- Just in case:
  samples = samples:float()
  gt = gt:float()

  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  model:training()
  model:cuda()

  -- Get model parameters:
  local params, gradParams = model:getParameters()
  -- Set optimization parameters:
  local optimState = {learningRate = 0.002}

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
    local gt_batches = gt_shuffled:view(bn, batch_size, 2)

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
        print('Loss = ' .. loss .. ', Norm = ' .. out_norm)

        return loss, gradParams
      end

      --- TODO: Check parameters for params optimState
      optim.sgd(feval, params, optimState)

      if (math.fmod(b, 10) == 0) then
        print('Training: epoch '.. epoch .. ' batch ' .. b .. ' points = ' .. (b*batch_size))
      end

    end

    if (math.fmod(epoch, 3) == 0) then
      optimState.learningRate = optimState.learningRate * 0.8
    end
    

  end

  return model

end



function M.evaluate(samples, model, batch_size)

  local n = samples:size(1)

  -- Set module to 'Evaluate' mode in contrast with training mode:
  model:evaluate()

  local ind_start = 1
  local ind_end = batch_size
  local outputs = torch.FloatTensor(n, 2)

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
