
%% Generate sampled point clouds + GT normals:
% CloudFromOFF('../data/shapes/151.off', 100000, 0, '../data/shapes/151_100k_0');
% CloudFromOFF('../data/shapes/326.off', 100000, 0, '../data/shapes/326_100k_0');
% CloudFromOFF('../data/shapes/332.off', 100000, 0, '../data/shapes/332_100k_0');
% CloudFromOFF('../data/shapes/359.off', 100000, 0, '../data/shapes/359_100k_0');
% CloudFromOFF('../data/shapes/364.off', 100000, 0, '../data/shapes/364_100k_0');
% CloudFromOFF('../data/shapes/367.off', 100000, 0, '../data/shapes/367_100k_0');
% CloudFromOFF('../data/shapes/066.off', 100000, 0, '../data/shapes/066_100k_0');
% CloudFromOFF('../data/shapes/centaur4.off', 100000, 0, '../data/shapes/centaur4_100k_0');


%% Run deep network (compiled c++ script):

nh_dir = '/home/yanir/Documents/Projects/DeepCloud/code/normals_HoughCNN/';
nh_exec = './HoughCNN_Exec';

% olddir = cd(nh_dir); 

model_num = '3';
%base_name = '151A_100k_0005';
base_name = 'fandisk100k';
postfix = '_CNN';

% Image size is based on model:
image_size = [33 33 3];
% hough_length = 33*33*3;

model_name = ['/home/yanir/Documents/Projects/DeepCloud/data/model_' model_num 's'];
input_name = ['/home/yanir/Documents/Projects/DeepCloud/data/shapes/' base_name '.xyz'];
% output_name = ['/home/yanir/Documents/Projects/DeepCloud/code/normals_HoughCNN/out/' base_name '_m' model_num '_out' postfix '.xyz'];
% output_name = ['/home/yanir/Documents/Projects/DeepCloud/data/out/' base_name '_normals_mynet.xyz'];
output_name = ['/home/yanir/Documents/Projects/DeepCloud/data/out/regression_model/fandisk100k_normals.xyz'];
% output_name = ['/home/yanir/Documents/Projects/DeepCloud/data/out/' base_name '_normals_pca.xyz'];
hough_name = ['/home/yanir/Documents/Projects/DeepCloud/code/normals_HoughCNN/out/' base_name '_m' model_num '_HoughAccum.mat'];

% cmd = [nh_exec ' -m ' model_name ' -i ' input_name ' -o ' output_name];
% 
% % eval(['!' cmd]);
% % status = system(cmd,'-echo');
% 
% display('Be a good boy and run this line for me, would you:');
% display(cmd);
% display('I will be waiting right here, champ.');
% % 
% % pause;
% % cd(olddir);

%% Evaluate results:
display('Displaying results...')
xyz = load(output_name);
pos_out = xyz(:, 1:3);
normals_out = xyz(:, 4:6);
n = size(xyz, 1);
% Orient normals such that the most significant value is always positive:
for i=1:n;
    if (max(abs(normals_out(i, :))) ~= max(normals_out(i, :)))
        normals_out(i, :) = -normals_out(i, :);
    end;
end;

% % Check consistency of normals:
% f = sum(abs(xyz(:, 4:6)), 2);
% figure; plot_function_pcd(xyz, f)

% d = xyz(:, 1:3);
% d(abs(d) < 1) = 0;
% g = sqrt(sum((xyz(:, 4:6) - d).^2));
% figure; plot_function_pcd(xyz, g)

normals_name = ['/home/yanir/Documents/Projects/DeepCloud/data/shapes/' base_name '.normals'];

%%%% Parsing hough accumulators from a text file:
% if (~exist('hough', 'var'))
%     fid = fopen(hough_name);
%     hough = zeros(n, hough_length);
%     i = 1;
%     
%     % Get one row of text from file:
%     hline = fgetl(fid);
%     while ischar(hline)
%         ts = textscan(hline, '%f');
%         hough(i, :) = ts{1};
%         if (mod(i, 100) == 0)
%             display(i);
%         end;
%         hline = fgetl(fid);
%         i = i + 1;
%     end;

%%%% Load hough accumulators from .mat file:
% x = load(hough_name);
% hough = x.hough;
% clear x;
% 

normals_gt = load(normals_name);
f = abs(sum(normals_out .* normals_gt, 2));
f(f>1) = 1;
f(1) = 0;
ang = rad2deg(acos(f));
% ang(ang > 15) = 15;
fig = figure('WindowStyle', 'docked'); 
% fig = figure;
plot_function_pcd(pos_out, ang); colorbar;
% dcm = datacursormode(fig);
% set(dcm, 'UpdateFcn', {@tooltip_vertex_image, pos_out, hough, image_size},'SnapToDataVertex','on');


% sorted = sort(ang, 'descend');
% 
% range = 1:10000;
% figure; plot(range, sorted(range));



% Create a histogram of errors:
% figure;
% histogram(f);

% f = sum((normals_out - normals_gt).^2, 2);
% figure; plot_function_pcd(xyz, f); colorbar;

% figure; plot_function_pcd(xyz, log(ang)); colorbar;


