%-- 21/11/16 01:00:44 pm --%
xyz = load('normals_HoughCNN/out.xyz');
n = length(xyz)
f = zeros(n, 1);
for i=1:n; f(i) = xyz(i, 1:3) * xyz(i, 4:6)'; end
f(1:100)
xyz(2, :)
find(xyz(:, 1) > 0.95 & xyz(:, 2) > 0.95)
xyz(424, :)
f = sum(xyz(:, 4:6) .^ 2, 2);
size(f)
f(1:100)
f(424)
f = sum(abs(xyz(:, 4:6)), 2);
f(424)
f(1:100)
plot_function_pcd(xyz(:, 1:3), abs(f))
xyz(424, :)
d = zeros(n, 3);
d = xyz;
d(abs(d) < 1) = 0;
d(424, :)
d = xyz(:, 1:3);
d(abs(d) < 1) = 0;
d(424, :)
g = zeros(n, 1);
for i=1:n; g(i) = d(i, :) * xyz(i, 4:6)'; end
g(400:450)
figure; plot_fuction_pcd(xyz, g)
figure; plot_function_pcd(xyz, g)
figure; plot_function_pcd(xyz, abs(g))
CloudFromOFF('../../data/shapes/151.off', 100000, 0, '../../data/shapes/151_100k_0.xyz');
CloudFromOFF('../../data/shapes/326.off', 100000, 0, '../../data/shapes/326_100k_0.xyz');
CloudFromOFF('../../data/shapes/332.off', 100000, 0, '../../data/shapes/332_100k_0.xyz');
CloudFromOFF('../../data/shapes/359.off', 100000, 0, '../../data/shapes/359_100k_0.xyz');
CloudFromOFF('../../data/shapes/364.off', 100000, 0, '../../data/shapes/364_100k_0.xyz');
CloudFromOFF('../../data/shapes/367.off', 100000, 0, '../../data/shapes/367_100k_0.xyz');
CloudFromOFF('../../data/shapes/66.off', 100000, 0, '../../data/shapes/066_100k_0.xyz');
CloudFromOFF('../../data/shapes/centaur4.off', 100000, 0, '../../data/shapes/centaur4_100k_0.xyz');
xyz = load('../normals_HoughCNN/out/332_100k_0_m1_out.xyz');
f = sum(abs(xyz(:, 4:6)), 2);
figure; plot_function_pcd(xyz, f)
xyz = load('../normals_HoughCNN/out/359_100k_0_m1_out.xyz');
f = sum(abs(xyz(:, 4:6)), 2);
figure; plot_function_pcd(xyz, f)
xyz = load('../normals_HoughCNN/out/centaur4_100k_0_m1_out.xyz');
f = sum(abs(xyz(:, 4:6)), 2);
figure; plot_function_pcd(xyz, f)
g = xyz(:, 4) - xyz(:, 5);
figure; plot_function_pcd(xyz, f)
g = xyz(:, 4) - xyz(:, 6);
g = xyz(:, 4) - xyz(:, 5);
figure; plot_function_pcd(xyz, g)
g = abs(xyz(:, 4) - xyz(:, 5));
figure; plot_function_pcd(xyz, g)
g = abs(xyz(:, 4)/2 + xyz(:, 5)/4 + xyz(:, 6)/8);
figure; plot_function_pcd(xyz, g)
xyz = load('../normals_HoughCNN/out/359_100k_0_m1_out.xyz');
xyz = load('../normals_HoughCNN/out/364_100k_0_m1_out.xyz');
f = sum(abs(xyz(:, 4:6)), 2);
figure; plot_function_pcd(xyz, g)
figure; plot_function_pcd(xyz, f)
xyz = load('../normals_HoughCNN/out/332_100k_0_m1_out.xyz');
d = xyz(:, 1:3);
d(abs(d) < 1) = 0;
d(1:100)
d(1:100, :)
xyz = load('../normals_HoughCNN/out/cube_100k_0_m1_out.xyz');
xyz = load('../normals_HoughCNN/out/cube100k_m1_out.xyz'');
xyz = load('../normals_HoughCNN/out/cube100k_m1_out.xyz');
f = sum(abs(xyz(:, 4:6)), 2);
figure; plot_function_pcd(xyz, f)
d = xyz(:, 1:3);
d(abs(d) < 1) = 0;
d(1:100, :)
d = 1 - abs(d);
d(1:100, :)
f = sum((xyz(:, 4:6).*d).^2, 2);
figure; plot_function_pcd(xyz, f)
f = sum(abs(xyz(:, 4:6).*d), 2);
figure; plot_function_pcd(xyz, f)
f = sum(abs(xyz(:, 4:6)).^0.3, 2);
figure; plot_function_pcd(xyz, f)
f = sum(abs(xyz(:, 4:6)).^0.5, 2);
figure; plot_function_pcd(xyz, f)
xyz = load('../normals_HoughCNN/out/332_100k_0_m1_out.xyz');
f = sum(abs(xyz(:, 4:6)).^0.5, 2);
figure; plot_function_pcd(xyz, f)
xyz = load('../normals_HoughCNN/out/359_100k_0_m1_out.xyz');
f = sum(abs(xyz(:, 4:6)).^0.5, 2);
figure; plot_function_pcd(xyz, f)
close all
%-- 22/11/16 12:09:18 pm --%
xyz = load('../normals_HoughCNN/out/332_100k_0_m1_out.xyz');
f = sum(abs(xyz(:, 4:6)).^0.5, 2);
figure; plot_function_pcd(xyz, f)
d = xyz(:, 1:3);
d(abs(d) < 1) = 0;
g = sqrt(sum((xyz(:, 4:6) - d).^2));
figure; plot_function_pcd(xyz, g)
dd = (xyz(:, 4:6) - d);
dd(1:100, :)
[d(1:50, :) xyz(1:50, 4:6)]
d(1:100, :)
d = xyz(:, 1:3);
d(1:100, :)
f = sum(abs(xyz(:, 4:6)), 2).^0.5;
figure; plot_function_pcd(xyz, g)
figure; plot_function_pcd(xyz, f)
f = sum(abs(xyz(:, 4:6)), 2).^0.1;
figure; plot_function_pcd(xyz, f)
f = sum(abs(xyz(:, 4:6)).^0.1, 2);
figure; plot_function_pcd(xyz, f)
max(f)
min(f)
f = sum(abs(xyz(:, 4:6)).^0.5, 2);
max(f)
min(f)
f = sum(abs(xyz(:, 4:6)), 2).^0.5;
max(f)
min(f)
f = sum(abs(xyz(:, 4:6)), 2).^10;
max(f)
min(f)
figure; plot_function_pcd(xyz, f)
xyz = load('../normals_HoughCNN/out/cube100k_m1_out.xyz');
d = xyz(:, 1:3);
d(1:100, :)
d(abs(d) < 1) = 0;
d(1:100, :)
g = sqrt(sum((xyz(:, 4:6) - d).^2));
g(1:100, :)
g(1:100)
g = sqrt(sum((xyz(:, 4:6) - d).^2, 2));
g(1:100, :)
g = sqrt(sum((abs(xyz(:, 4:6)) - abs(d)).^2, 2));
g(1:100, :)
figure; plot_function_pcd(xyz, g)
xyz = load('../normals_HoughCNN/out/cube100k_m1_out.xyz');
pcd = xyz(:, 1:3);
pcd = pcd + noise*(rand(size(pcd))-1/2)*(max(max(pcd)-min(pcd)));
noise = 0.01;
pcd = pcd + noise*(rand(size(pcd))-1/2)*(max(max(pcd)-min(pcd)));
dlmwrite('cube100k_noise.xyz', pcd, 'precision', '%.6f', 'delimiter', ' ');
xyz = load('../normals_HoughCNN/out/cube100k_noise_m1_out.xyz');
f = sum(abs(xyz(:, 4:6)).^0.5, 2);
figure; plot_function_pcd(xyz, g)
figure; plot_function_pcd(xyz, f)
f = sum(abs(xyz(:, 4:6)), 2);
figure; plot_function_pcd(xyz, f)
close all
%-- 23/11/16 04:13:03 pm --%
A = rand(100000, 1089);
whos
ind = randperm(1000, 100000)
ind = randperm(100000, 1000)
A(ind, :)
ind = randperm(100000, 100)
A(ind, 1:10)
clear all
h = load('normals_HoughCNN/out/cube100k_HoughAccum.txt', 'ascii');
fid = fopen('normals_HoughCNN/out/cube100k_HoughAccum.txt');
h = textscan(fid, '%f');
h
h = reshape(h{1}, [100000, 1089]);
h(1, :)
h = reshape(h, 108900000, 1);
h(1:10)
h([1 1090])
h([1 1089])
h([1 100000])
h([1 100001])
h(12)
sum(h(1:11))
h = reshape(h, 1089, 100000)';
h(1, :)
save('normals_HoughCNN/out/cube100k_HoughAccum', 'h')
close(fid)
fclose(fid)
fprintf(pFileAccum, "\n");
cube = load('normals_Hoh ughCNN/out/cube
fid = fopen('normals_HoughCNN/out/cube100k_HoughAccum.txt');
h = textscan(fid, '%f');
fclose(fid);
h = reshape(h{1}, 1089, 100000)';
h(1, 1:10)
h(1, 1:12)
h(1, 1:100)
sum(sum(h))
sum(h(1, :))
sum(h(2, :))
sum(h(3, :))
save('normals_HoughCNN/out/cube100k_HoughAccum', 'h')
xyz = load('normals_HoughCNN/out/cube100k_m1_out.xyz');
xyz(1, :)
xyz(2, :)
ind = find(sum(xyz, 2) > 2.9)
ind = find(sum(xyz, 2) > 2.99)
ind(1)
ind(2)
h2 = sum(h, 2);
h2(1:21)
A = [h2 xyz];
A(1:21, :)
fprintf(pFileAccum, "\n");
2 = sum(h, 2);
ind = find(sum(xyz(:, 1:3), 2) > 2.9)
h2(ind)
A(ind, :)
ind = find(sum(xyz(:, 1:3), 2) > 2.97)
A(ind, :)
plot(h(ind, :))
plot(h(ind, :)')
h(27706, :)
sum(h(ind, :))
sum(h(ind, :), 2)
h(ind(2), :)
sum(h(ind, :), 2)
h(ind(4), :)
sum(h(ind, :), 2)
max(h(ind, :), 2)
max(h(ind, :), [], 2)
ind2 = find(sum(xyz(:, 1:3), 2) < 0.2)
ind2 = find(sum(xyz(:, 1:3), 2) < -2.97)
ind2 = find(sum(xyz(:, 1:3), 2) < -2.95)
A(ind, :)
h(ind(1:2), :)
sum(h)
ind = find(xyz(:, 1) == 1);
sum(h(ind, :))
plot(sum(h(ind, :)))
ind2 = find(xyz(:, 2) == 1);
ind3 = find(xyz(:, 3) == 1);
figure; plot(sum(h(ind, :))); hold on; plot(sum(h(ind2, :))); plot(sum(h(ind3, :))); legend('x', 'y', 'z')
hx = sort(sum(h(ind, :)))
plot(hx)
hy = sort(sum(h(ind2, :)))
hz = sort(sum(h(ind3, :)));
plot(hy)
plot(hz)
h(123, :)
min(h(h > 0))
max(h(:)
max(h(:))
xyz(2, :)
d = xyz(:, 1:3) - repmat(xyz(2, 1:3), 100000, 1);
[min_val, min_id] = min(sum(d.^2))
sum(d.^2)
[min_val, min_id] = min(sum(d.^2, 2))
[min_val, min_id] = min(sum(d(3:100000).^2, 2))
[min_val, min_id] = min(sum(d(3:100000, :).^2, 2))
xyz([2 18005], :)
xyz([2 18007], :)
h([2 18007], :)
xyz([2 18007], :)
reshape(h(2, :), 33, 33)
reshape(h(18007, :), 33, 33)
reshape(h(12345, :), 33, 33)
xyz(12345, :)
xyz(54312, :)
reshape(h(54312, :), 33, 33)
d = xyz(:, 1:3) - repmat(xyz(54312, 1:3), 100000, 1);
d(d == 0) = 100;
[min_val, min_id] = min(sum(d.^2, 2))
xyz([54312 94563], :)
d = xyz(:, 1:3) - repmat(xyz(54312, 1:3), 100000, 1);
[min_val, min_id] = min(sum(d.^2, 2))
s = sort(sum(d.^2, 2));
s(1:10)
[s, si] = sort(sum(d.^2, 2));
si(1:10)
xyz([54312 69169], :)
s(2)
h([54312 69169], :)
reshape(h([54312], :), 33, 33)
reshape(h([69169], :), 33, 33)
reshape(h([69196], :), 33, 33)
d = xyz(:, 1:3) - repmat(xyz(69196, 1:3), 100000, 1);
[s, si] = sort(sum(d.^2, 2));
s(1:10)
si(1:10)
xyz(si(1:2), :)
reshape(h([69196], :), 33, 33)
reshape(h([32945], :), 33, 33)
h(si(1:4), :)
norm(h(s1, :) - h(s2, :))
norm(h(s(1), :) - h(s(2), :))
norm(h(si(1), :) - h(si(2), :))
norm(h(si(1), :) - h(si(210), :))
norm(h(si(1), :) - h(si(10), :))
B = rand(5, 3)
dist(B)
clear B
dh = h(i, :) - h(12345, :);
dh = h - repmat(h(12345, :), 100000, 1);
sh = sum(dh.^2, 2);
min(sh)
min(sh(sh>0))
[~, mi] = min(sh(sh>0))
xyz([12345 70417], :)
h([12345 70417], :)
norm(h(12345, :) - h(mi, :))
[sh, shi] = sort(sum(dh.^2, 2));
sh(1:10)
shi(1:10)
h([12345 70418], :)
norm(h(12345, :)
norm(h(12345, :))
norm(h(70417, :))
xyz([12345 70417], :)
reshape(h([12345], :), 33, 33)
imshow(reshape(h([12345], :), 33, 33))
imagesc(reshape(h([12345], :), 33, 33))
fid = fopen('normals_HoughCNN/out/cube100k_HoughAccum.txt');
h = textscan(fid, '%f');
fclose(fid);
h = reshape(h{1}, [100000, 1089]);
h([12345 70418], :)
dh = h - repmat(h(12345, :), 100000, 1);
[sh, shi] = sort(sum(dh.^2, 2));
sh(1:10)
shi(1:10)
xyz([12345 79435], :)
imagesc(reshape(h([12345], :), 33, 33))
imagesc(reshape(h([79435], :), 33, 33))
figure; imagesc(reshape(h([12345], :), 33, 33))
xyz(1, :)
dh = h - repmat(h(1, :), 100000, 1);
[sh, shi] = sort(sum(dh.^2, 2));
shi(1:10)
xyz(shi(1:10), :)
d = xyz(:, 1:3) - repmat(xyz(1, 1:3), 100000, 1);
[s, si] = sort(sum(d.^2, 2));
s(1:3)
si(1:3)
xyz(si(1:10), :)
h(si(1:2), :)
figure; imagesc(reshape(h(si(1), :), 33, 33))
figure; imagesc(reshape(h(si(2), :), 33, 33))
xyz(1:10, :)
[(1:100)' xyz(1:100, :)]
h(94, :)
imagesc(reshape(h([94], :), 33, 33))
figure; imagesc(reshape(h([94], :), 33, 33))
reshape(h([94], :), 33, 33)
)
reshape(h([94], :), 33, 33)
find(xyz(:, 1) > 0.99 & xyz(:, 2) > 0.99)
xyz(1662, :)
xyz(99671, :)
figure; imagesc(reshape(h([1662], :), 33, 33))
figure; imagesc(reshape(h([99671], :), 33, 33))
hh = load('normals_HoughCNN/out/cube100k_HoughAccum.txt');
GT = xyz(1:3, :);
GT(abs(GT)<1) = 0;
GT(1:100, :)
GT
GT = xyz(:, 1:3);
GT(abs(GT)<1) = 0;
GT(1:100, :)
err_deg = sum(abs(GT .* xyz(:, 4:6)), 2);
err_deg(1:100)
err_deg(2)
xyz(2, :)
normals = xyz(:, 4:6);
for i=1:100000; normals(i, :) = normals(i, :) / norm(normals(i, :)); end;
err_deg = sum(abs(GT .* normals), 2);
err_deg(1:100)
norm(normals(1, :))
normals(1, :)
normals(1, 2)
normals(1, 1)
normals(1, 1) ^2
norm(normals(1, :)) - 1
normals(1, 2) ^ 2
normals(1, 3) ^ 2
sum(normals(1, :) .^ 2)
sum(normals(1, :) .^ 2) - 1
normals(1, :) .^ 2
nn = normals(1, :) .^ 2
nn(1)
sum(nn)
sum(nn) - 1
nn(3)
nn(1) + nn(3)
whois
whos
nn(1) + nn(3)
nn(1) + n(2) + nn(3)
nn(1) + nn(2) + nn(3)
nn(1) + nn(2) + nn(3) - 1
nn(1) + nn(2) + nn(3) - nn(2)
nn(2) - 1
err_deg = sum(abs(GT - normals), 2);
err_deg = sum(abs(GT .* normals), 2);
err_dist = sum(abs(GT - normals), 2);
err_dist(1:100)
err_dist = sum((abs(GT) - abs(normals)).^2, 2);
err_dist(1:100)
hist(err_deg)
figure; hist(err_dist)
max(err_deg)
min(err_deg)
h = hist(err_dist);
h
loglog(h)
h = hist(err_deg);
h
loglog(h)
fid = fopen('normals_HoughCNN/out/cube100k_HoughAccum.txt');
h = reshape(h{1}, [100000, 1089]);
h = textscan(fid, '%f');
fclose(fid);
h = reshape(h{1}, [100000, 1089]);
