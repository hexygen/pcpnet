require 'cutorch'
require 'sys'

local M={};

-- Quick and dirty knn search: finds K nearest neighbors of a tensor where each row 
-- is a point in Euclidean space.
-- This is a basic and not very efficient algorithm. It takes about 30-40 seconds for a set of 100,000 points.
-- Can be replaced with a wrapper for FLANN or some equivalent in the future.
function M.searchknn(data, k) 

  print('Computing ' .. k .. ' nearest neighbors...')
  sys.tic()

  local n = data:size(1)
  local d = data:size(2)

  -- Allocate tensors for results:
  local knn = torch.LongTensor(n, k)
  local dist = torch.FloatTensor(n, k)
  dist:fill(10000)



  -- Trying a faster (maybe) algorithm?

  -- First, search for nearest neighbors of first point:
  local diff_1 = data - data[1]:repeatTensor(1, n)
  local d1 = torch.norm(diff_1, 2, 2):squeeze()
  --local d1 = diff_1:pow(2):sum(2):squeeze()

  -- sort distances to first point:
  local sort1, si1 = d1:sort()

  -- top k neighbors of first point are the first k values:
  dist[1] = sort1[{{1, k}}]
  knn[1] = si1[{{1, k}}]

  local band = torch.FloatTensor(k*2, d)
  local di = torch.FloatTensor(k*2, 1)

  -- For the next rows, find nearest neighbors in a band around them 
  -- according to distance from first point:
  for i=2,n do

    local ii = si1[i]
    local bandwidth = k*2 -- so starting bandwidth total is k*4 on both sides
    local start_low_band = math.max(1, i-bandwidth+1)
    local end_low_band = math.max(1, i-1)
    local start_high_band = math.min(n, i+1)
    local end_high_band = math.min(n, i+bandwidth-1)

    local done = false

    while not done do

      local low_band = si1[{{start_low_band, end_low_band}}]
      local high_band = si1[{{start_high_band, end_high_band}}]
      local band_ind = low_band:cat(high_band, 1)
      band:index(data, 1, band_ind)

--      -- Four rows in one - perhaps it is more efficient?
--      band:index(data, 1, 
--          si1[{{start_low_band, end_low_band}}]:cat(si1[{{start_high_band, end_high_band}}], 1))

      band:add(-1, data[ii]:view(1, d):expandAs(band))
      di = torch.norm(band, 2, 2)
      --di:sum(band:cmul(band), 2) -- cmul(self) is faster than pow(2)!

      dist_k, ind_k = di:squeeze():topk(k)
      local max_val = dist_k:max()
      dist[ii] = dist_k
      -- Put si1(band_ind(ind_k)) into knn[ii]:
      knn[ii]:index(band_ind, 1, ind_k)

      -- Check whether bandwidth has to grow:
      local d1_min = d1[si1[start_low_band]]
      local d1_max = d1[si1[end_high_band]]
      local d1_ii = d1[ii]

      if (((start_low_band == 1) or (max_val < d1_ii - d1_min)) and ((end_high_band == n) or (max_val < d1_max - d1_ii))) then
        -- Found neighbors are close enough so any other point cannot be closer:
        done = true
      else
        -- It is possible that points outside the band are closer - check bands outside:
        bandwidth = bandwidth*2
        --end_low_band = math.max(1, start_low_band-1)
        start_low_band = math.max(1, i-bandwidth+1) -- using new bandwidth

        --start_high_band = math.min(n, end_high_band+1)
        end_high_band = math.min(n, i+bandwidth-1) -- using new bandwidth

--        print('i = ' .. i .. ', time = ' .. sys.toc() .. ', bandwidth = ' .. bandwidth .. '\n')
      end

    end

    if (math.fmod(i, n/20) == 0) then
      print(string.format('%d%%\t%.2f seconds', i/n*100, sys.toc()))

    end
  end


  -- Sort nearest neighbors lists:
  dist, sort_ind = dist:sort(2)
  -- Change index of knn according to distance:
  for i=1,n do
    knn[i] = knn[i]:index(1, sort_ind[i])
  end


  local dt = sys.toc()
  print('Computed nearest neighbors in ' .. dt .. ' seconds.')


  return knn, dist
end



return M;