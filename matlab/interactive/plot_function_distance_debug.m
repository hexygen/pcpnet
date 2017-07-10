function plot_function_distance_debug(shape, f, D, gt)
    fig3 = figure(1);
    trimesh(shape.TRIV, shape.X, shape.Y, shape.Z, gt, ...
        'EdgeColor', 'interp', 'FaceColor', 'interp');
    h = datacursormode(fig3);
    set(h,'UpdateFcn',{@debug_gt,shape,gt},'SnapToDataVertex','on');
    title('Ground truth symmetry');
    axis equal;
    axis off;
    
    fig = figure(2);
    trimesh(shape.TRIV, shape.X, shape.Y, shape.Z, f, ...
        'EdgeColor', 'interp', 'FaceColor', 'interp');
    h = datacursormode(fig);
    set(h,'UpdateFcn',{@debug_dists,shape,D,gt},'SnapToDataVertex','on');
    title('Computed symmetry map');

    axis equal;
    axis off;
end