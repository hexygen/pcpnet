function [txt] = tooltip_vertex_image(~, event_obj, vertex_pos, vertex_data, image_size)
    curr_pos = get(event_obj, 'Position');
    
    C = bsxfun(@minus, vertex_pos, curr_pos); 
    dists = sum(C.^2,2);
    [~, vid] = min(dists);
% 	vid = find(dists == min(dists));

    img = reshape(vertex_data(vid, :), image_size);
    figure(123); image(img); title(['vertex ' num2str(vid)]);

    txt = num2str(vid);
end
