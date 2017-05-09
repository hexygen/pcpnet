require 'cutorch'
require 'sys'

local M={};

function M.evaluate(hough, batch_size)

  local n = hough:size(1)
  -- Make sure hough data is cuda:
  hough = hough:cuda()
  
  -- Set module to 'Evaluate' mode in contrast with training mode:
  model:evaluate()

  local ind_start = 1
  local ind_end = batch_size
  local normals = torch.FloatTensor(n, 2)

  while (ind_start < n) do

    if (ind_end > n) then
      ind_end = n
    end
    
    -- forward
    local chunk = hough[{{ind_start, ind_end}}]
    local normals_chunk = model:forward(chunk)
    
    normals[{{ind_start, ind_end}, {}}] = normals_chunk:float()
    
    print('Forward step batch ' .. ind_end .. ' in ' .. sys.toc() .. ' seconds.')
    
    ind_start = ind_start + batch_size
    ind_end = ind_end + batch_size
    
  end

  return normals
end
  

return M;
