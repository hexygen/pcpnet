function h = show_matrix(h, m, varargin)

nvargs = struct( ...
    'precision',4, ...
    'format','%0.4f');

if numel(h) ~= numel(m)+1
    h = gobjects(numel(m)+1,1);
end

[nvargs,nvmask] = nvpairs2struct(varargin, nvargs, false, 0);
varargin(nvmask) = [];

h(1) = pcolor(m([1:end,1],[1:end,1]));
% set(temp,'EdgeColor','none');
set(h(1),varargin{:});
[x,y] = meshgrid(1:size(m,2),1:size(m,1));
if ~isempty(nvargs.format)
    valstrings = num2str(m(:),nvargs.format);
else
    valstrings = num2str(m(:),nvargs.precision);
end
valstrings = strtrim(cellstr(valstrings));
% valstrings_norm = num2str(100.*cm(:)./sum(cm(:)),'%0.2f');
% valstrings_norm = strtrim(cellstr(valstrings_norm));
% valstrings_norm = cellfun(@(x) ['(',x,'%)'],valstrings_norm,'UniformOutput',false);
% htext = text(x(:)+0.5,y(:)+0.35,valstrings(:),'HorizontalAlignment','center');
% htext = text(x(:)+0.5,y(:)+0.5,valstrings(:),'HorizontalAlignment','center');
h(2:end) = text(x(:)+0.5,y(:)+0.5,valstrings(:),'HorizontalAlignment','center');
% htext2 = text(x(:)+0.5,y(:)+0.65,valstrings_norm(:),'HorizontalAlignment','center');
% imagesc(cm);

% ax = gca;
% % ax.CLim = [0,max(max(cm_RAID(:)),max(cm_SingleSC(:)))];
% ax.CLim = [0,1];
% ax.YDir = 'reverse';
% ax.XTick = 1.5:size(cm,2)+1;
% ax.YTick = 1.5:size(cm,1)+1;
% ax.XTickLabel = classnames;
% ax.YTickLabel = classnames;
% ax.XAxisLocation = 'top';
% xlabel('predicted','FontSize',14);
% ylabel('actual','FontSize',14);
% rotateXLabels(ax,30);
% colorbar;
% fig = gcf;
% fig.Position = [0,0,500,400];

% hold off;
