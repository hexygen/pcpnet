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

  local rp = torch.FloatTensor(n, k)
  local col = torch.FloatTensor(n, 1);
  -- Draw first column (no constraints):
  rp:select(2, 1):rand(n):mul(max):ceil()
  
  for i=2,k do
    col:rand(n):mul(max):ceil()
    
    -- Re-draw numbers that are the same as previous columns:
    local prev = rp[{{},{1,i-1}}]
    col_expand = col:view(n, 1):expandAs(prev)
    
    mask = torch.eq(prev, col_expand):sum(2)
    
    while mask:any() do
      local new_rand = torch.rand(mask:sum()):mul(max):ceil():float()
      col:maskedCopy(mask, new_rand)
    
      -- Update mask (col_expand is a view on col so it is updated):
      mask = torch.eq(prev, col_expand):sum(2)
    end
    
        -- Put column in permutations matrix:
    rp[{{}, {i}}] = col
  end
  
  
  return rp
end

function M.findlast(str,char)
  local ind = str:reverse():find(char,1,true)
  
  if ind then
    ind = str:len() - (ind-1)
  end

  return ind
end

function M.splitext(filename)

  local sep = package.config:sub(1,1)
  local sep_ind = M.findlast(filename,sep)
  local dot_ind = M.findlast(filename,".")
  local n = nil
  local e = nil
  if dot_ind and dot_ind > sep_ind then
    n = filename:sub(1, dot_ind-1)
    e = filename:sub(dot_ind)
  else
    n = filename
    e = ''
  end

  return n, e
end

return M
