require 'cutorch'

local M={};


-- Checks if file exists:
function M.exists(name)
  local f=io.open(name,"r")
  if f~=nil then 
    io.close(f) 
    return true 
  else 
    return false 
  end
end


-- Batch randperm - quickly create a matrix where every row is a partial random permutation.
-- n = number of rows
-- k = number of columns
-- max = maximum number to permute from
function M.randperm_partial(n, k, max)

  local rp = torch.Tensor(n, k)
  local col = torch.Tensor(n, 1);
  -- Draw first column (no constraints):
  rp:select(2, 1):rand(n):mul(max):ceil()
  
  for i=2,k do
    col:rand(n):mul(max):ceil()
    
    -- Re-draw numbers that are the same as previous columns:
    local prev = rp[{{},{1,i-1}}]
    col_expand = col:view(n, 1):expandAs(prev)
    
    mask = torch.eq(prev, col_expand):sum(2)
    
    while mask:any() do
      local new_rand = torch.rand(mask:sum()):mul(max):ceil()
      col:maskedCopy(mask, new_rand)
    
      -- Update mask (col_expand is a view on col so it is updated):
      mask = torch.eq(prev, col_expand):sum(2)
    end
    
        -- Put column in permutations matrix:
    rp[{{}, {i}}] = col
  end
  
  
  return rp
end

return M