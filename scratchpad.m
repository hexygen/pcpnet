%%
% pts_filename = '../data/shapes/cube100k.xyz';
% gt_filename = '../data/shapes/cube100k.normals';
% % normals_filename = '../data/out/regression_model/cube100k_normals_boulch.xyz'; % boulch
% % normals_filename = '../data/out/regression_model/cube100k_normals.xyz'; % ours regression model
% normals_filename = '../data/out/classification_model/cube100k_normals.xyz'; % ours classification model
% % normals_filename = '../data/out/pca_only/cube100k_normals.xyz'; % pca only

pts_filename = '../data/shapes/fandisk100k.xyz';
gt_filename = '../data/shapes/fandisk100k.normals';
% normals_filename = '../data/out/regression_model/fandisk100k_normals_boulch.xyz'; % boulch
% normals_filename = '../data/out/regression_model/fandisk100k_normals.xyz'; % ours regression model
normals_filename = '../data/out/classification_model/fandisk100k_normals.xyz'; % ours classification model
% normals_filename = '../data/out/classification_model/fandisk100k_normals_avg.xyz'; % ours classification model
% normals_filename = '../data/out/pca_only/fandisk100k_normals.xyz'; % pca only


% load points
pts = importdata(pts_filename);

% load normals
gt = importdata(gt_filename);
normals = importdata(normals_filename);
normals = normals(:,4:6);

% load pca
ind = strfind(normals_filename,'_normals');
if isempty(ind)
    error('Cannot parse normal filename.');
end
pca_filename = [normals_filename(1:ind-1),'_pca_100.h5'];
pcas = h5read(pca_filename,'/pcas');

% re-normalize normals
gt = gt ./ sqrt(sum(gt.^2,2));
normals = normals ./ sqrt(sum(normals.^2,2));

gt_local = zeros(size(gt));
normals_local = zeros(size(normals));
for i=1:size(normals,1)
    normals_local(i,:) = (pcas(:,:,i) * normals(i,:)')';
    gt_local(i,:) = (pcas(:,:,i) * gt(i,:)')';
end

% % angle error in degrees
% normal_error = abs(rad2deg(acos(abs(dot(normals',gt')'))));
% disp('mean absolute normal angle error (in degrees):')
% disp(nanmean(normal_error));

% same as loss during training
batch_size = 64; % must be the same value used during training for accurate results
normal_error = min(...
    nansum((normals_local(:,1:2) - gt_local(:,1:2)).^2,2),...
    nansum((-normals_local(:,1:2) - gt_local(:,1:2)).^2,2));
mean_batch_normal_error = sum(normal_error) * (batch_size/size(normals_local,1));
disp('mean batch normal error:')
disp(mean_batch_normal_error);

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
set(gca,'CLim',[0,0.1]); % for training loss
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
