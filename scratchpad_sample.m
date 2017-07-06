%%
addpath('sample');

%% point clouds without noise
CloudFromOFF('../data/shapes/sources/cube.off',100000,'../data/shapes/cube',0);
% CloudFromOFF('../data/shapes/sources/fandisk.off',100000,'../data/shapes/fandisk100k',0);
% CloudFromOFF('../data/shapes/sources/bunny.off',100000,'../data/shapes/bunny100k');
% CloudFromOFF('../data/shapes/sources/armadillo.off',100000,'../data/shapes/armadillo100k');
% CloudFromOFF('../data/shapes/sources/dragon.off',100000,'../data/shapes/dragon100k');
% CloudFromOFF('../data/shapes/sources/happy.off',100000,'../data/shapes/happy100k');

%% point clouds with brown noise
% CloudFromOFF('../data/shapes/sources/cube.off',100000,'../data/shapes/cube100k_noise_brown_3e-2',0.03,'brown');
% CloudFromOFF('../data/shapes/sources/fandisk.off',100000,'../data/shapes/fandisk100k_noise_brown_3e-2',0.03,'brown');
% CloudFromOFF('../data/shapes/sources/bunny.off',100000,'../data/shapes/bunny100k_noise_brown_3e-2',0.03,'brown');
% CloudFromOFF('../data/shapes/sources/armadillo.off',100000,'../data/shapes/armadillo100k_noise_brown_3e-2',0.03,'brown');
CloudFromOFF('../data/shapes/sources/dragon.off',100000,'../data/shapes/dragon100k_noise_brown_3e-2',0.03,'brown');
CloudFromOFF('../data/shapes/sources/happy.off',100000,'../data/shapes/happy100k_noise_brown_3e-2',0.03,'brown');

%%
% temp = colored_noise([1000,1000],-2,0.02);
temp = colored_noise([500,500,500],-2,0.03);

%%
figure;
% image(temp,'CDataMapping','scaled');
image(temp(:,:,2),'CDataMapping','scaled');

%% create dictionary of template patches
% cube has 100000 points in an area of 6 (edge length is 1), so point density is 100000/6
% hough transform has 100 points, so the neighborhood has an area of ~ 100 / (100000/6) = 0.006 (disc with radius ~ 0.043)
% patch 1: 0.1*0.1 * 1 = 0.01 => 167 points to get equal density
% patch 2: 0.05*0.1 * 2 = 0.01 => 167 points to get equal density
% patch 3: 0.05^2 * 3 = 0.0075 => 125 points to get equal density

CloudFromOFF('../data/shapes/sources/patch_1.off',167,'../data/shapes/patch_1');
CloudFromOFF('../data/shapes/sources/patch_2.off',167,'../data/shapes/patch_2');
CloudFromOFF('../data/shapes/sources/patch_3.off',125,'../data/shapes/patch_3');
