% fit point cloud patches to the local neighborhood of each point in a point cloud
% shape is n x 3 array of points
% patches is cell array containing is n_i x 3 arrays of points
function [e,t,n,shape_subsample_inds] = pc_fit_patches(shape_samples,patch_verts,patch_faces,varargin)
    
    nvargs = struct(...
        'shape_sample_fraction',1,...
        'knn_indices',[],...
        'sample_density',100000/6);
    nvargs = nvpairs2struct(varargin,nvargs);
    
    % samples patches with given sample density
    patch_pc = cell(1,numel(patch_verts));
    patch_pca = cell(1,numel(patch_verts));
    patch_centroid = zeros(numel(patch_verts),3);
    patch_samples = cell(1,numel(patch_verts));
    patch_sample_count = zeros(1,numel(patch_verts));
    patch_face_normals = cell(1,numel(patch_verts));
    for i=1:numel(patch_verts)
        patch_area = sum(computeArea(patch_verts{i}, patch_faces{i}));
        patch_sample_count(i) = round(nvargs.sample_density * patch_area / 4);
        if patch_sample_count(i) < 10
            error('Too few samples, patch area or sample density might be too small.');
        end
        [patch_samples{i}, ~, ~] = sample_mesh(patch_verts{i}, patch_faces{i}, patch_sample_count(i));
        patch_centroid(i,:) = mean(patch_samples{i},1);
        patch_samples{i} = patch_samples{i} - patch_centroid(i,:);
        
        % face normals
        patch_face_normals{i} = cross(...
            patch_verts{i}(patch_faces{i}(:,2),:) - patch_verts{i}(patch_faces{i}(:,1),:),...
            patch_verts{i}(patch_faces{i}(:,3),:) - patch_verts{i}(patch_faces{i}(:,1),:));
        % normalize normal:
        patch_face_normals{i} = patch_face_normals{i} ./ sqrt(sum(patch_face_normals{i}.^2,2));
        % orient it such that the most significant value is always positive:
        mask = max(abs(patch_face_normals{i}),[],2) ~= max(patch_face_normals{i},[],2);
        patch_face_normals{i}(mask,:) = -patch_face_normals{i}(mask,:);
        
        patch_pc{i} = pointCloud(patch_samples{i});
        
        patch_pca{i} = pca(patch_samples{i});
        if det(patch_pca{i}) < 0
            patch_pca{i}(:,end) = -patch_pca{i}(:,end); % flip smallest principal component
        end
        patch_pca{i} = eye(3); % temp
    end
    
    if nvargs.shape_sample_fraction < 1
        shape_subsample_count = max(1,round(size(shape_samples,1) * nvargs.shape_sample_fraction));
        shape_subsample_inds = randsample(size(shape_samples,1),shape_subsample_count);
        shape_subsample_inds = sort(shape_subsample_inds,'ascend');
    else
        shape_subsample_inds = 1:size(shape_samples,1);
    end
    
    % get k nearest neighbors for each point on the shape
    shape_patch_sample_count = round(mean(patch_sample_count));
    if ~isempty(nvargs.knn_indices)
        knn_idx = nvargs.knn_indices;
    else
        [knn_idx,~] = knnsearch(shape_samples,shape_samples(shape_subsample_inds,:),'K',shape_patch_sample_count);
    end
    
    rots = cat(3,eye(3),roty(90),roty(180),roty(270),rotz(90),rotz(270));
    rots = repmat(rots,1,1,1,4);
    for i=1:size(rots,3)
        rots(:,:,i,2) = rots(:,:,i,2) * rotx(90);
        rots(:,:,i,3) = rots(:,:,i,3) * rotx(180);
        rots(:,:,i,4) = rots(:,:,i,4) * rotx(270);
    end
    rots = reshape(rots,3,3,[],1);
    
    ticid = tic;
    % fit each patch to the neighborhood around each shape sample and save
    % best transform, error and normal at shape point for best fit
    e = zeros(numel(shape_subsample_inds),1);
    t = zeros(numel(shape_subsample_inds),16);
    n = zeros(numel(shape_subsample_inds),3);
    for i=1:numel(shape_subsample_inds)
        
        if mod(i,5) == 1
            elapsed_time = toc(ticid);
            disp([num2str(i),' / ',num2str(numel(shape_subsample_inds)),' - ETA: ',num2str(elapsed_time * ((numel(shape_subsample_inds)-(i-1)) / (i-1))),' seconds'])
        end
        
        shape_patch_samples = shape_samples(knn_idx(i,:),:);
        shape_patch_centroid = mean(shape_patch_samples,1);
        shape_patch_samples = shape_patch_samples - shape_patch_centroid; % origin at centroid
        shape_patch_pc = pointCloud(shape_patch_samples);
        shape_patch_pca = pca(shape_patch_samples);
        if det(shape_patch_pca) < 0
            shape_patch_pca(:,end) = -shape_patch_pca(:,end); % flip smallest principal component
        end
        shape_patch_pca = eye(3); % temp
        
        
        patch_transform = cell(1,numel(patch_samples));
        patch_error = zeros(1,numel(patch_samples));
        for j=1:numel(patch_samples)
            for k=1:size(rots,3)
                [patch_transform{j},~,patch_error(j)] = pcregrigid(patch_pc{j},shape_patch_pc,...
                    'InitialTransform',affine3d([rots(:,:,k) * shape_patch_pca * patch_pca{j}',[0;0;0]; 0,0,0,1]),...
                    'Tolerance',[0.0001,0.009],...
                    'MaxIterations',100);
            end
        end
        
        [e(i),patch_ind] = min(patch_error);
        t(i,:) = patch_transform{patch_ind}.T(:)';
        
        % transform patch geometry, get nearest face on patch to the
        % current point
        patch_verts_t = patch_verts{patch_ind} - patch_centroid(patch_ind,:);
        patch_verts_t = [patch_verts_t,ones(size(patch_verts{patch_ind},1),1)]; %#ok<AGROW> % add homogeneous coordinate
        patch_verts_t = (patch_transform{patch_ind}.T' * patch_verts_t')';
        patch_verts_t = patch_verts_t(:,1:3) ./ patch_verts_t(:,4) + shape_patch_centroid; % re-homogenize
        
        patch_face_normals_t = (patch_transform{patch_ind}.T(1:3,1:3)' * patch_face_normals{patch_ind}')';
        
        d = pointTriangleDistance3D_mex(shape_samples(shape_subsample_inds(i),:)',...
            patch_verts_t(patch_faces{patch_ind}(:,1),:)',...
            patch_verts_t(patch_faces{patch_ind}(:,2),:)',...
            patch_verts_t(patch_faces{patch_ind}(:,3),:)');
        [~,face_ind] = min(d);
        
        n(i,:) = patch_face_normals_t(face_ind,:);
    end
    
end
