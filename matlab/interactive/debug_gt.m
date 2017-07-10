function [txt] = debug_gt(obj, event_obj, shape, gt)
    pos = get(event_obj, 'Position');
    
    C = bsxfun(@minus,[shape.X shape.Y shape.Z], pos); 
    dists = sum(C.^2,2);
	vid = find(dists == min(dists));

    txt = num2str(gt(vid));
end
