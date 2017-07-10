function [txt] = debug_dists(obj, event_obj, shape, D, gt)
    pos = get(event_obj, 'Position');
    
    C = bsxfun(@minus,[shape.X shape.Y shape.Z], pos); 
    dists = sum(C.^2,2);
	vid = find(dists == min(dists));
    
    figure(3);

    plot(D(vid,:)); hold on;
    title(sprintf('Distances to the samples for vertex %d', vid));
    yL = get(gca,'YLim');
    plot([gt(vid) gt(vid)], yL);
    xlabel('fps sample');
    ylabel(sprintf('D(%d, fps sample)', vid));
    hold off;

    [~,md] = min(D(vid,:));
    
    txt = [num2str(gt(vid)) ' ' num2str(md)];
end
