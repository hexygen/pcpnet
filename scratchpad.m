%%
addpath('sample');

%%
shapedir = '../data/shapes';
normaldir = '../data/out';

% model_names = { ...
%     'pca_only'; ...
%     'boulch'; ...
%     'regression_model'; ...
%     'classification_model'; ...
%     };

model_names = { ...
    'temp'; ...
    };

model_variants = { ...
    ''; ... % no noise
    ...'_noise_brown_3e-2'; ... % brown noise
    };

method_variants = { ...
    ''; ...
    '_cube'; ... % no noise
    ...'_avg'; ...
    };

% shape_names = { ...
%     'cube100k'; ...
% 	'fandisk100k'; ...
%     'bunny100k'; ...
%     'armadillo100k'; ...
%     'dragon100k'; ...
%     'happy100k'; ...
%     };

shape_names = { ...
    'fandisk100k'; ...
    };

shape_variants = { ...
    ''; ... % no noise
    ...'_noise_brown_3e-2'; ... % brown noise
    };

batch_size = 64; % must be the same value used during training (just so it is comparable)

[method_vi,model_vi,model_ni] = ndgrid(1:numel(method_variants),1:numel(model_variants),1:numel(model_names));
[shape_ni,shape_vi] = ndgrid(1:numel(shape_names),1:numel(shape_variants));
normal_error_loss = nan(numel(shape_ni),numel(model_ni));
normal_error_angle = nan(numel(shape_ni),numel(model_ni));

table_method_names = cell(1,numel(model_ni));
table_shape_names = cell(1,numel(shape_ni));
for s=1:numel(shape_ni)
    disp(['shape ',num2str(s),' / ',num2str(numel(shape_ni))]);
    
    shape_name = shape_names{shape_ni(s)};
    shape_variant = shape_variants{shape_vi(s)};
    table_shape_names{s} = [shape_name, shape_variant];
    
    pts_filename = fullfile(shapedir,[shape_name,shape_variant,'.xyz']);
    gt_filename = fullfile(shapedir,[shape_name,shape_variant,'.normals']);
    pca_filename = [pts_filename(1:end-4),'_pca_33_1000.h5'];
    
    % load points and ground truth normals
    gt = importdata(gt_filename);
    gt = gt ./ sqrt(sum(gt.^2,2)); % re-normalize normals
    pcas = h5read(pca_filename,'/pcas');
    
    gt_local = zeros(size(gt));
    for i=1:size(gt,1)
        gt_local(i,:) = (pcas(:,:,i) * gt(i,:)')';
    end

    for m=1:numel(model_ni)
        disp(['method ',num2str(m),' / ',num2str(numel(model_ni))]);
    
        model_name = model_names{model_ni(m)};
        model_variant = model_variants{model_vi(m)};
        method_variant = method_variants{method_vi(m)};
        table_method_names{m} = [model_name, model_variant, method_variant];
    
        normals_filename = fullfile(normaldir,[model_name,model_variant],[shape_name,shape_variant,'_normals',method_variant,'.xyz']);    
        
        if exist(pts_filename,'file') == 2 && exist(gt_filename,'file') == 2 && exist(normals_filename,'file')

            % load estimated normals
            normals = importdata(normals_filename);
            normals = normals(:,4:6);
            normals = normals ./ sqrt(sum(normals.^2,2)); % re-normalize normals
            
            normals_local = zeros(size(normals));
            for i=1:size(normals,1)
                normals_local(i,:) = (pcas(:,:,i) * normals(i,:)')';
            end

            % angle error in degrees
            ne_angle = abs(rad2deg(acos(abs(dot(normals',gt')'))));

            % same as loss during training
            ne_loss = min(...
                nansum((normals_local(:,1:2) - gt_local(:,1:2)).^2,2),...
                nansum((-normals_local(:,1:2) - gt_local(:,1:2)).^2,2));
            
            normal_error_loss(s,m) = nanmean(ne_loss) * batch_size;
            normal_error_angle(s,m) = nanmean(ne_angle);
        else
            warning(['Could not find all files for combination: ',model_name,' - ',model_variant,' - ',method_variant,' - ',shape_name,' - ',shape_variant]);
        end
    end
end

normal_error_loss = array2table(normal_error_loss,'VariableNames',matlab.lang.makeValidName(table_method_names),'RowNames',table_shape_names);
normal_error_angle = array2table(normal_error_angle,'VariableNames',matlab.lang.makeValidName(table_method_names),'RowNames',table_shape_names);

normal_error_loss_vis = normal_error_loss{:,:};
mask = all(isnan(normal_error_loss_vis),1);
normal_error_loss(:,mask) = [];
normal_error_angle(:,mask) = [];

%%
figure;
m = flipud(normal_error_loss{:,:});

show_matrix([],m,'EdgeColor',[0.9,0.9,0.9]);
set(gca,'CLim',[0,0.1]);

%%
figure;
uitable('Data',normal_error_loss{:,:},'ColumnName',normal_error_loss.Properties.VariableNames,...
    'RowName',normal_error_loss.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);



%%
% model_name = 'pca_only';
% model_name = 'boulch';
% model_name = 'regression_model';
% model_name = 'classification_model';
model_name = 'temp';

model_variant = ''; % no noise
% model_variant = '_noise_brown_3e-2'; % brown noise

method_variant = '_cube';
% method_variant = '_avg';

% shape_name = 'cube100k';
shape_name = 'fandisk100k';
% shape_name = 'bunny100k';
% shape_name = 'armadillo100k';
% shape_name = 'dragon100k';
% shape_name = 'happy100k';

shape_variant = ''; % no noise
% shape_variant = '_noise_brown_3e-2'; % brown noise

batch_size = 64; % must be the same value used during training (just so it is comparable)

pts_filename = fullfile(shapedir,[shape_name,shape_variant,'.xyz']);
gt_filename = fullfile(shapedir,[shape_name,shape_variant,'.normals']);
normals_filename = fullfile(normaldir,[model_name,model_variant],[shape_name,shape_variant,'_normals',method_variant,'.xyz']);    
pca_filename = [pts_filename(1:end-4),'_pca_33_1000.h5'];


% disp('mean absolute normal angle error (in degrees):')
% disp(nanmean(normal_error_angle));

[pts,gt,normals,normal_error_loss,normal_error_angle] = load_result(pts_filename,gt_filename,normals_filename,pca_filename);

disp('mean batch normal error:')
disp(nanmean(normal_error_loss) * batch_size);

% normal_error = normal_error_angle;
normal_error = normal_error_loss;

%%
normal_error_compare = normal_error;

%% pcas
x = pcas(1,:,:);
y = pcas(2,:,:);
z = pcas(3,:,:);
figure;
% pca_vis = scatter3(pts(:,1),pts(:,2),pts(:,3),20,permute(z,[3,2,1]).*0.5+0.5,'.','MarkerFaceColor','flat');
pca_vis = scatter3(pts(:,1),pts(:,2),pts(:,3),20,abs(permute(x,[3,2,1])),'.','MarkerFaceColor','flat');
set(gca,'Clipping','off');
% set(gca,'CLim',[0,90]); % for angle error
% set(gca,'CLim',[0,0.2]); % for training loss
xlabel('X');
ylabel('Y');
zlabel('Z');
set(gcf,'Color','white');
% set(gca,'Visible','off');
axis equal;
set(gcf,'Name',normals_filename);
view(110,-30)

%% gt normals
figure;
gt_vis = scatter3(pts(:,1),pts(:,2),pts(:,3),1,abs(gt),'.');
set(gca,'Clipping','off');
xlabel('X');
ylabel('Y');
zlabel('Z');
set(gcf,'Color','white');
axis equal;

%% computed normals
figure;
normals_vis = scatter3(pts(:,1),pts(:,2),pts(:,3),1,abs(normals),'.');
set(gca,'Clipping','off');
xlabel('X');
ylabel('Y');
zlabel('Z');
set(gcf,'Color','white');
axis equal;

%% errors
figure;

error_vis = scatter3(pts(:,1),pts(:,2),pts(:,3),20,normal_error,'.','MarkerFaceColor','flat');
% error_vis = scatter3(pts(:,1),pts(:,2),pts(:,3),20,ne_our - ne_b,'.','MarkerFaceColor','flat');
set(gca,'Clipping','off');
% set(gca,'CLim',[0,45]); % for angle error
set(gca,'CLim',[0,0.2]); % for training loss
xlabel('X');
ylabel('Y');
zlabel('Z');
set(gcf,'Color','white');
set(gca,'Visible','off');
axis equal;
set(gcf,'Name',normals_filename);
view(110,-30)

%% error improvement over another method
figure;

error_vis = scatter3(pts(:,1),pts(:,2),pts(:,3),20,normal_error_compare - normal_error,'.','MarkerFaceColor','flat');
% error_vis = scatter3(pts(:,1),pts(:,2),pts(:,3),20,ne_our - ne_b,'.','MarkerFaceColor','flat');
set(gca,'Clipping','off');
% set(gca,'CLim',[0,45]); % for angle error
set(gca,'CLim',[-0.1,0.1]); % for training loss
xlabel('X');
ylabel('Y');
zlabel('Z');
set(gcf,'Color','white');
set(gca,'Visible','off');
axis equal;
set(gcf,'Name',normals_filename);
view(110,-30)

%%
% 90396
disp(find(error_vis.BrushData));
disp(normal_error(find(error_vis.BrushData)));

%% analyze hough histograms output of the classification network
% hough_opt_filename = '../data/out/classification_model/cube100k_hough_opt.h5';
hough_opt_filename = '../data/out/classification_model/fandisk100k_hough_opt.h5';
hough_opt = h5read(hough_opt_filename,'/hough_opt');

%%
inds = find(error_vis.BrushData);
figure;
imshow(imresize(reshape(hough_opt(:,inds(1)),33,33,1).*100,[330,330],'nearest'));

%%
% cube has 100000 points in an area of 6 (edge length is 1), so point density is 100000/6
% hough transform has 100 points, so the neighborhood has an area of ~ 100 / (100000/6) = 0.006 (disc with radius ~ 0.043)
% patch 1: 0.1*0.1 * 1 = 0.01 => 167 points to get equal density
% patch 2: 0.05*0.1 * 2 = 0.01 => 167 points to get equal density
% patch 3: 0.05^2 * 3 = 0.0075 => 125 points to get equal density
sample_density = 100000/6; 

shape_filename = '../data/shapes/cube100k.xyz';

patch_filenames = {...
    '../data/shapes/sources/patch_1.off',...
    '../data/shapes/sources/patch_2.off',...
    '../data/shapes/sources/patch_3.off',...
    };

shape_samples = dlmread(shape_filename);

patch_verts = cell(1,numel(patch_filenames));
patch_faces = cell(1,numel(patch_filenames));
for i=1:numel(patch_filenames)
    [patch_verts{i},patch_faces{i}] = readoffmesh(patch_filenames{i});
    patch_verts{i} = patch_verts{i} .* 2;
end

[patch_fitting_error,patch_transform,shape_subsample_normals,shape_subsample_inds] = pc_fit_patches(...
    shape_samples,patch_verts,patch_faces,'sample_density',sample_density,'shape_sample_fraction',0.1);

save('../data/test_patchfit_baseline.mat','patch_fitting_error','patch_transform','shape_subsample_normals','shape_subsample_inds');


%%
                [patch_transform{j},~,patch_error(j)] = pcregrigid(patch_pc{j},shape_patch_pc,...
                    'InitialTransform',affine3d([rots(:,:,k) * shape_patch_pca' * patch_pca{j},[0;0;0]; 0,0,0,1]));
%%
                [patch_transform{j},patch_pc_t,patch_error(j)] = pcregrigid(patch_pc{j},shape_patch_pc,...
                    'InitialTransform',affine3d([rots(:,:,k) * shape_patch_pca * patch_pca{j}',[0;0;0]; 0,0,0,1]),'Verbose',true);
%%
j = patch_ind;

%%
%                 [patch_transform{j},patch_pc_t,patch_error(j)] = pcregrigid(patch_pc{j},shape_patch_pc,...
%                     'InitialTransform',affine3d([rots(:,:,k) * shape_patch_pca * patch_pca{j}',[0;0;0]; 0,0,0,1]),'Verbose',true,'Tolerance',[0.0001,0.009],'MaxIterations',100);

patch_samples_t = [patch_pc{j}.Location,ones(size(patch_pc{j}.Location,1),1)]; % add homogeneous coordinate
patch_samples_t = (patch_transform{j}.T' * patch_samples_t')';
patch_samples_t = patch_samples_t(:,1:3) ./ patch_samples_t(:,4); % re-homogenize

% patch_samples_t = (patch_transform{j}.T(1:3,1:3) * patch_pc{j}.Location')';

% patch_samples_t = patch_pc_t.Location;

patch_samples_t2 = (rots(:,:,k) * shape_patch_pca * patch_pca{j}' * patch_pc{j}.Location')';
% patch_samples_t2 = (rots(:,:,k) * shape_patch_pca' * patch_pca{j} * patch_pc{j}.Location')';

figure;
hold on;
% scatter3(patch_pc{j}.Location(:,1),patch_pc{j}.Location(:,2),patch_pc{j}.Location(:,3),'c.');
% scatter3(patch_samples{j}(:,1),patch_samples{j}(:,2),patch_samples{j}(:,3),'c.');
scatter3(patch_samples_t2(:,1),patch_samples_t2(:,2),patch_samples_t2(:,3),'g.');
scatter3(patch_samples_t(:,1),patch_samples_t(:,2),patch_samples_t(:,3),'r.');
scatter3(shape_patch_pc.Location(:,1),shape_patch_pc.Location(:,2),shape_patch_pc.Location(:,3),'bx');
set(gca,'Clipping','off');
axis equal;
hold off;


%%
nvec = [shape_samples(shape_subsample_inds(i),:);shape_samples(shape_subsample_inds(i),:) + n(i,:) * 0.1];

figure;
hold on;
patch('Vertices',patch_verts_t,'Faces',patch_faces{patch_ind},'FaceAlpha',0.5,'FaceColor','red');
scatter3(shape_samples(knn_idx(i,:),1),shape_samples(knn_idx(i,:),2),shape_samples(knn_idx(i,:),3),'bx');
line(nvec(:,1),nvec(:,2),nvec(:,3),'Color','green');
set(gca,'Clipping','off');
axis equal;
hold off;
