function [txt] = tooltip_vertex(~, event_obj, shape, vertex_data)
    pos = get(event_obj, 'Position');
    
    C = bsxfun(@minus,[shape.X shape.Y shape.Z], pos); 
    dists = sum(C.^2,2);
    [~, vid] = min(dists);
% 	vid = find(dists == min(dists));

    txt = num2str(vertex_data(vid));
end
