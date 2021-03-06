

nh_dir = '/home/yanir/Documents/Projects/DeepCloud/code/normals_HoughCNN/';
nh_exec = './HoughCNN_Exec';

% olddir = cd(nh_dir); 

model_num = '1';
%base_name = '151A_100k_0005';
base_name = 'cube100k';
% postfix = '_CNN';
postfix = '';

% Image size is based on model:
image_size = [33 33 3];
hough_length = 33*33*3;

model_name = ['/home/yanir/Documents/Projects/DeepCloud/data/model_' model_num 's'];
input_name = ['/home/yanir/Documents/Projects/DeepCloud/data/shapes/' base_name '.xyz'];
% shape_name_boulch = ['/home/yanir/Documents/Projects/DeepCloud/code/normals_HoughCNN/out/' base_name '_m' model_num '_out' postfix '.xyz'];
% shape_name_boulch = ['/home/yanir/Documents/Projects/DeepCloud/data/out/' base_name '_normals_boulch.xyz'];
% shape_name_09 = ['/home/yanir/Documents/Projects/DeepCloud/data/out/' base_name '_normals_mynet.xyz'];
% shape_name_08 = ['/home/yanir/Documents/Projects/DeepCloud/data/out/old/' base_name '_normals_mynet.xyz'];
% shape_name_old = ['/home/yanir/Documents/Projects/DeepCloud/data/out/old/cube100k_normals_mynet_old3.xyz'];
% shape_name_pca = ['/home/yanir/Documents/Projects/DeepCloud/data/out/' base_name '_normals_pca.xyz'];
hough_name = ['/home/yanir/Documents/Projects/DeepCloud/code/normals_HoughCNN/out/' base_name '_m' model_num '_HoughAccum.mat'];
gt_cube = '/home/yanir/Documents/Projects/DeepCloud/data/shapes/cube100k.normals';
gt_fandisk = '/home/yanir/Documents/Projects/DeepCloud/data/shapes/fandisk100k.normals';
gt_bunny = '/home/yanir/Documents/Projects/DeepCloud/data/shapes/bunny100k.normals';
gt_armadillo = '/home/yanir/Documents/Projects/DeepCloud/data/shapes/armadillo100k.normals';
gt_dragon = '/home/yanir/Documents/Projects/DeepCloud/data/shapes/dragon100k.normals';
gt_happy = '/home/yanir/Documents/Projects/DeepCloud/data/shapes/happy100k.normals';

shape_path = '/home/yanir/Documents/Projects/DeepCloud/data/out/';

shapes = {'re3/cube100k_normals.xyz', 're3/cube', gt_cube;
          're3/fandisk100k_normals.xyz', 're3/fandisk', gt_fandisk;
          're3/bunny100k_normals.xyz', 're3/bunny', gt_bunny;
          're3/armadillo100k_normals.xyz', 're3/armadillo', gt_armadillo;
          're3/dragon100k_normals.xyz', 're3/dragon', gt_dragon;
          're3/happy100k_normals.xyz', 're3/happy', gt_happy;
          're4/cube100k_normals.xyz', 're4/cube', gt_cube;
          're4/fandisk100k_normals.xyz', 're4/fandisk', gt_fandisk;
          're4/bunny100k_normals.xyz', 're4/bunny', gt_bunny;
          're4/armadillo100k_normals.xyz', 're4/armadillo', gt_armadillo;
          're4/dragon100k_normals.xyz', 're4/dragon', gt_dragon;
          're4/happy100k_normals.xyz', 're4/happy', gt_happy;};
% shapes = {'regression_model/fandisk100k_normals.xyz', 'Fandisk->Fandisk', gt_fandisk;
%           'regression_model/cube_fandisk100k_normals.xyz', 'Cube->Fandisk', gt_fandisk;
%           'pca_only/fandisk100k_normals.xyz', 'PCA->Fandisk', gt_fandisk;
%           'regression_model/bunny100k_normals.xyz', 'Fandisk->Bunny', gt_bunny;
%           'regression_model/cube_bunny100k_normals.xyz', 'Cube->Bunny', gt_bunny;
%           'pca_only/bunny100k_normals.xyz', 'PCA->Bunny', gt_bunny;
%           'regression_model/armadillo100k_normals.xyz', 'Fandisk->Armadillo', gt_armadillo;
%           'regression_model/cube_armadillo100k_normals.xyz', 'Cube->Armadillo', gt_armadillo;
%           'pca_only/armadillo100k_normals.xyz', 'PCA->Armadillo', gt_armadillo;};
          
% shapes = {[base_name '_normals_noflat.xyz'], 'no flat';
%           [base_name '_normals_mynet.xyz'], 'no flat 3 epochs';
%           [base_name '_normals_boulch.xyz'], 'boulch'; 
%           [base_name '_normals_09.xyz'], 'band 09';
%           ['old/' base_name '_normals_mynet.xyz'], 'band 08';
%           [base_name '_normals_pca.xyz'], 'pca'};
% Evaluate results:

e = [];
p = [];
for i=1:size(shapes, 1)
    [err, ang] = EvaluateError([shape_path shapes{i,1}], shapes{i, 3});
    [e(:, i), p(:, i)] = PrecisionCurve(ang);
end;

figure;
plot(e, p);
legend(shapes(:, 2));

% figure;
% plot(e(:, 1:5), p(:, 1:5));
% legend(shapes(1:5, 2));
% 
% figure;
% plot(e(:, 4:6), p(:, 4:6));
% legend(shapes(4:6, 2));
% 
% figure;
% plot(e(:, 7:9), p(:, 7:9));
% legend(shapes(7:9, 2));


% figure; plot(eb, pb, e08, p08, e09, p09, ep, pp, eo, po); legend('boulch', 'ours 08', 'ours 09', 'pca', 'ours old');

% range = 1:100000;
% figure; plot(range, sort(ang_boulch, 'descend'), range, sort(ang_mynet, 'descend'), range, sort(ang_pca, 'descend')); legend('boulch', 'ours', 'pca');



