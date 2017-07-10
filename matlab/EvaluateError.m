function [ err, ang, pos_out, normals_out ] = EvaluateError( shape_name, gt_name )
% Evaluates the error of an output model compared to the ground truth
% model.

display(['Evaluating shape ' shape_name '...']);
xyz = load(shape_name);
pos_out = xyz(:, 1:3);
normals_out = xyz(:, 4:6);
n = size(xyz, 1);
% Orient normals such that the most significant value is always positive:
for i=1:n;
    if (max(abs(normals_out(i, :))) ~= max(normals_out(i, :)))
        normals_out(i, :) = -normals_out(i, :);
    end;
end;

normals_gt = load(gt_name);
err = abs(sum(normals_out .* normals_gt, 2));
err(err>1) = 1;
ang = rad2deg(acos(err));


end

