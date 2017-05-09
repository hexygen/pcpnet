require 'cutorch'
require 'sys'
local utils = require('utils')
local knnlib = require('knn')

local M={};

local RAND_CHUNK_SIZE = 100000

local randperm
local rp_ind
local function randperm_init(k)
  randperm = utils.randperm_partial(RAND_CHUNK_SIZE, 3, k):long()
  rp_ind = 1
end

local function get_randperm(chunk, k)
    -- Get a chunk of random triangles:
    local rp_end = rp_ind + chunk - 1
    if (randperm:size(1) < rp_end) then
      -- Get a new chunk:
      randperm_init(k)
      -- Update rp_end accordingly:
      rp_end = chunk
    end
  
    local ri = randperm[{{rp_ind, rp_end}}]
    
    rp_ind = rp_end + 1
    return ri
end

-------------------------------------------------------------------------
-- Performs non-anisotropic hough transform on unsuspecting point clouds.
-- data = unsuspecting point cloud in question
-- k = number of nearest neighbors to sample
-- num_of_samples = number of samples to use for each point on the shape
-- hist_size = length and width of the histogram for the final output
-- TODO: Add support for anisotropic hough transform
-- TODO: Add support for multiscale networks
function M.hough(data, k, num_of_samples, hist_size) 
  print('Computing hough tranform...')

  local n = data:size(1)
  local d = data:size(2)
  local hs2 = hist_size*hist_size

  local knn, dist

  -- Generate permutations of 3 vertices out of the first 100 nearest neigbors:
  sys.tic()
  randperm_init(k)
  print('Computed randperm_partial in '.. sys.toc() .. ' seconds.')

  sys.tic()

--  if utils.exists('test_knn.dat') and utils.exists('test_knn_dist.dat') then
--    -- Load knn from cache files:
--    knn = torch.load('test_knn.dat')
--    dist = torch.load('test_knn_dist.dat')
--  else
    -- Compute knn and save in cache files:
    knn, dist = knnlib.searchknn(data, k)

--    torch.save('test_knn.dat', knn)
--    torch.save('test_knn_dist.dat', dist)
--  end


  local hough = torch.FloatTensor(n, hs2)
  local pcas = torch.FloatTensor(n, 3, 3)

  local normals = torch.FloatTensor(num_of_samples, 2)
  -- example normal = 0.1761    0.4402    0.8805  
  normals:select(2, 1):fill(0.1761)
  normals:select(2, 2):fill(0.4402)
--  normals:select(2, 3):fill(0.8805)

  local v1 = torch.FloatTensor(k, 3)
  local v2 = torch.FloatTensor(k, 3)
  local v3 = torch.FloatTensor(k, 3)

  local di = torch.FloatTensor(k, 3)
  local di_rot = torch.FloatTensor(k, 3)
  local normals_rot = torch.FloatTensor(num_of_samples, 2)

  local pca1u, pca1s, pca1v
  local pca2u, pca2s, pca2v
  
  for i=1,n do
    --- Compute hough transform for one point ---

    -- data is now local k nearest neighbors of i:
    di:index(data, 1, knn[i])

    -- Computing non-anisotropic PCA:
    -- (note: output values are initialized on first call and then reused)
    if (pca1u and pca1s and pca1v) then
      M.pca(di, pca1u, pca1s, pca1v)
    else
      pca1u, pca1s, pca1v = M.pca(di)
    end
    


    -- Rotate data using pca (result goes into di) (transposed since the order is reversed):
    di_rot:mm(di, pca1u:t())

    -- Get samples of random triangles:
    local ri = get_randperm(num_of_samples, k)

    -- Get vertices from ids:
    local r1 = ri[{{}, {1}}]:squeeze()
    local r2 = ri[{{}, {2}}]:squeeze()
    local r3 = ri[{{}, {3}}]:squeeze()
    v1:index(di_rot, 1, r1)
    v2:index(di_rot, 1, r2)
    v3:index(di_rot, 1, r3)
    
    -- Compute vectors (v2 - v1) and (v3 - v1):
    v2:add(-1, v1)
    v3:add(-1, v1)
    -- Cross product:
    local normals = v2:cross(v3)
    
    -- Reverse normals if their last component is negative:
    local ind = torch.gt(normals[{{}, {3}}], 0):double()
    -- change ind from [0, 1] to [-1, 1] to multiply normals:
    ind:add(-0.5):mul(2)
    normals:cmul(ind:expandAs(normals))

    -- Normalize normals:
    normals:cdiv(normals:norm(2, 2):expandAs(normals))

    normals = normals[{{}, {1,2}}]

    -- Compute 2D PCA, of the two first coordinates of the normal:
    -- (note: output values are initialized on first call and then reused)
    if (pca2u and pca2s and pca2v) then
      M.pca(normals, pca2u, pca2s, pca2v)
    else
      pca2u, pca2s, pca2v = M.pca(normals)
    end

    -- Rotate normals by 2D pca:
    normals_rot:mm(normals, pca2u:t())
    
    -- Update pca matrix so the rotation back would be correct:
    -- (note: pca2u comes before, unless they are trnasposed as used above)
    pca1u[{{1,2}, {1,2}}] = torch.mm(pca2u, pca1u[{{1,2}, {1,2}}])

    -- Create histogram image by quantizing results:
    -- Histogram image is hist_size*hist_size cells:
    local c1 = normals:select(2, 1)
    local c2 = normals:select(2, 2)
    -- c = ((c+1)/2) * hist_size:
    c1:add(1):div(2):mul(hist_size):floor():clamp(0, hist_size-1)
    c2:add(1):div(2):mul(hist_size):floor():clamp(0, hist_size-1)

    -- computing histogram:
    hough[i] = torch.histc(torch.add(c1, hist_size, c2), hs2, 0, hs2)
    -- Saving pca for point:
    pcas[i] = pca1u


    if (math.fmod(i, n/100) == 0) then
      print(string.format('%d%%\t%.2f seconds', i/n*100, sys.toc()))

    end

  end


  print('Computed Hough transform of all points in ' .. sys.toc() .. ' seconds.')

  return hough, pcas
end

-- Performs anisotropic hough transform on unsuspecting point clouds.
-- k = number of nearest neighbors to sample
function M.hough_aniso(data, k) 

  print('Computing hough tranform...')
  sys.tic()

  local n = data:size(1)
  local d = data:size(2)

  local knn, dist = knnlib.searchknn(v, k)

  -- Area of influence is actually the distance to the 5th nearest neighbor
  -- (according to code from Boulch et al)
  local area = dist:select(2, 5)

  -- computing weighted (anisotropic) PCA:
  local pca = M.pca_weighted(data, area)


  --[[
  TO BE CONTINUED!
  ]]



  for i=1,n do
    if (math.fmod(i, n/20) == 0) then
      print(string.format('%d%%\t%.2f seconds', i/n*100, sys.toc()))

    end

  end


  print('Computed nearest neighbors in ' .. sys.toc() .. ' seconds.')


  --return ???, ???
end

function M.pca(x, out_u, out_s, out_v)
  -- Centering data around mean:
  local mean = torch.mean(x, 1)
  local xm = x - mean:expandAs(x)
  -- Since we are only taking eigenvectors, we don't need to compute xm = xm / sqrt(n-1):
  -- xm:div(math.sqrt(x:size(1)-1))

  -- Using SVD of centered data instead of eigenvalues of covariance:
  -- local w,s,v = torch.svd(xm:t())
  if (out_u and out_s and out_v) then
    torch.svd(out_u, out_s, out_v, xm:t())
  else
    local u,s,v = torch.svd(xm:t())
    return u,s,v
  end

end

function M.pca_weighted(x, weights)
  -- Multiply points by weights:
  local xw = torch.cmul(x, weights:expandAs(x))
  -- Compute weighted mean:
  local mean = xw:sum(1)/area:sum(1)
  local xm = x - mean 

  -- Weighted convariance:
  local xmw = torch.cmul(xm, weights:expandAs(xm))
  -- cov = (weights * xm)' * xm:
  local cov = xmw:t():mm(xm)

  -- PCA is eigenvectors of covariance matrix:
  local _, v = torch.symeig(cov, 'V')

  return v
end


-- Translate the results of the deep net to a 3D normal by 
-- computing the third coordinate and applying pca to each row
function M.postprocess_normals(normals, pcas)
  local n = normals:size(1)
  -- Compute 3D normals from 2D normals:
  local norms_2d = torch.norm(normals, 2, 2):pow(2):squeeze()
  -- Calculate 1 - norm_2d:
  local normals_3 = torch.FloatTensor(n, 1)
  normals_3:fill(1):add(-1, norms_2d):clamp(0, 1):sqrt()

  normals = torch.cat(normals, normals_3, 2)

  -- Rotate normals back using PCA:
  sys.tic()
  print('Rotating normals...')

  -- Adding a dimension for batch operation:
  normals:resize(n, 1, 3)

  -- Batch multiplying matrices with normal vectors:
  -- (note: pcas should be inverted and transposed. Since these are 
  --        rotation matrices the inverse cancels the transpose)
  normals = torch.bmm(normals:double(), pcas)

  normals:resize(n, 3)

  print('Rotated normals in ' .. sys.toc() .. ' seconds.')

  return normals
end


return M;