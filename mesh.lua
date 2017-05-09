require 'cutorch'
require 'sys'

local M={};

function M.readOff(filename) 

  print('Reading file ' .. filename .. '...')
  sys.tic()
  -- local t = os.clock()

  -- Opens a file in read mode
  local file = assert(io.open(filename, 'r'))

  -- Reads the first line from the file:
  local str = file:read()

  if string.sub(str, 1, 3) ~= 'OFF' then
    error('The file is not a valid OFF one.')
  end

  -- Read number of vertices:
  local nv, nf = file:read('*n', '*n')

  -- Move to the next line (discard third number in header):
  file:read()


  -- Allocate tensor for vertices:
  local v = torch.FloatTensor(nv, 3)
  -- Read vertices
  for i = 1, nv do

    local x, y, z = file:read('*n', '*n', '*n')
    v[{i, {}}] = torch.FloatTensor({x, y, z})

  end

  -- Allocate tensor for vertices:
  local f = torch.FloatTensor(nf, 3)
  -- Read faces
  for i = 1, nf do

    -- Discard first number in the row (number of vertices in face, always 3):
    local _, x, y, z = file:read('*n', '*n', '*n', '*n')
    f[{i, {}}] = torch.FloatTensor({x, y, z})

  end

  -- closes the opened file
  file:close()


--  local dt = os.difftime(os.time(), t)
  local dt = sys.toc()
  print('File ' .. filename .. ' read in ' .. dt .. ' seconds.')

  -- Create shape structure:
  local shape = {}
  shape.vertices = v
  shape.faces = f

  return shape
end


function M.readXYZ(filename) 

  print('Reading file ' .. filename .. '...')
  sys.tic()

  -- Opens a file in read mode
  local file = assert(io.open(filename, 'r'))

  -- Read entire file:
  local strall = file:read('*a')

  -- Find number of numbers ('words') in file:
  local _, nv = strall:gsub('%S+', '')
  nv = nv / 3

  -- Allocate tensor for vertices:
  local v = torch.FloatTensor(nv, 3)

  -- Fill tensor with data from string:
  local pos = 1
  v:apply(function()
      -- Find next number:
      local strnum = string.match(strall, "%S+", pos)
      -- Advance read position in string:
      pos = pos + strnum:len() + 1
      -- Convert string to number and apply to Tensor:
      return tonumber(strnum)
    end)


  -- closes the opened file
  file:close()

  local dt = sys.toc()
  print('File ' .. filename .. ' read in ' .. dt .. ' seconds.')

  -- Return vertices tensor:
  return v
end



function M.writeXYZ(filename, v, normals)
  print('Writing file ' .. filename .. '...')
  sys.tic()

  local line_length = 3
  
  if (normals) then
    v = torch.cat(v, normals, 2)
    line_length = 6
  end
  

  -- Opens a file in read mode
  local file = assert(io.open(filename, 'w'))

  -- Write each vertex to the file:
  local ind = 1
  v:apply(function(x)
      -- Write the value to file:
      file:write(string.format('%.6f', x))
      
      -- Advance indicator:
      ind = ind + 1
      if (ind > line_length) then
        -- Move to next line:
        ind = 1
        file:write('\n')
      else
        -- Write a space between columns:
        file:write(' ')
      end
      
    end)


  -- closes the opened file
  file:close()

  local dt = sys.toc()
  print('File ' .. filename .. ' written in ' .. dt .. ' seconds.')

  -- Return vertices tensor:
  return 
end



return M;