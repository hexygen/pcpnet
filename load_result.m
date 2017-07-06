function [pts,gt,normals,normal_error_loss,normal_error_angle] = load_result(pts_filename,gt_filename,normals_filename,pca_filename)

    % load points and ground truth normals
    pts = importdata(pts_filename);
    gt = importdata(gt_filename);
    gt = gt ./ sqrt(sum(gt.^2,2)); % re-normalize normals

    % load estimated normals
    normals = importdata(normals_filename);
    normals = normals(:,4:6);
    normals = normals ./ sqrt(sum(normals.^2,2)); % re-normalize normals

    % load pca

    pcas = h5read(pca_filename,'/pcas');

    gt_local = zeros(size(gt));
    normals_local = zeros(size(normals));
    for i=1:size(normals,1)
        normals_local(i,:) = (pcas(:,:,i) * normals(i,:)')';
        gt_local(i,:) = (pcas(:,:,i) * gt(i,:)')';
    end

    % angle error in degrees
    normal_error_angle = abs(rad2deg(acos(abs(dot(normals',gt')'))));

    % same as loss during training
    normal_error_loss = min(...
        nansum((normals_local(:,1:2) - gt_local(:,1:2)).^2,2),...
        nansum((-normals_local(:,1:2) - gt_local(:,1:2)).^2,2));

end
