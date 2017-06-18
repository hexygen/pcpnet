%%
pts_filename = '../data/shapes/cube100k.xyz';
gt_filename = '../data/shapes/cube100k.normals';
% normals_filename = '../data/out/cube100k_normals_mynet.xyz'; % ours
normals_filename = '../data/out/model2/cube100k_normals_mynet.xyz'; % ours model 2
% normals_filename = '../data/out/cube100k_normals_boulch.xyz'; % boulch
% normals_filename = '../data/out/cube100k_normals_yanir_mynet.xyz'; % boulch

pts = importdata(pts_filename);
gt = importdata(gt_filename);
normals = importdata(normals_filename);
normals = normals(:,4:6);
% re-normalize normals
gt = gt ./ sqrt(sum(gt.^2,2));
normals = normals ./ sqrt(sum(normals.^2,2));

% normal_error = 1-abs(dot(normals',gt')');
normal_error = rad2deg(acos(abs(dot(normals',gt')')));

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
set(gca,'Clipping','off');
set(gca,'CLim',[0,90]);
xlabel('X');
ylabel('Y');
zlabel('Z');
set(gcf,'Color','white');
set(gca,'Visible','off');
axis equal;

%%
find(error_vis.BrushData)