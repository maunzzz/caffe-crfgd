function setup_voc_data
%% SETUP
%download voc2012 and set path
voc_base_dir = '/media/cvia/disk2/Data/test_pascal/VOCdevkit/VOC2012';
create_edge_gt = true;

%% STUFF
fprintf('Writing list files for images and labels\n');
fprintf('Saving labels as raw png\n');
save_as_raw_png(fullfile(voc_base_dir,'SegmentationClass'), fullfile(voc_base_dir,'ImageSets','Segmentation','train.txt'), fullfile(voc_base_dir,'SegmentationRaw'));
save_as_raw_png(fullfile(voc_base_dir,'SegmentationClass'), fullfile(voc_base_dir,'ImageSets','Segmentation','val.txt'), fullfile(voc_base_dir,'SegmentationRaw'));
fprintf('Writing list files for images and labels\n');
write_list(fullfile(voc_base_dir,'JPEGImages'),fullfile(voc_base_dir,'ImageSets','Segmentation','train.txt'),fullfile('..','training','data_lists','train_images.txt'),'.jpg');
write_list(fullfile(voc_base_dir,'SegmentationRaw'),fullfile(voc_base_dir,'ImageSets','Segmentation','train.txt'),fullfile('..','training','data_lists','train_labels.txt'),'.png');
write_list(fullfile(voc_base_dir,'JPEGImages'),fullfile(voc_base_dir,'ImageSets','Segmentation','val.txt'),fullfile('..','training','data_lists','val_images.txt'),'.jpg');
write_list(fullfile(voc_base_dir,'SegmentationRaw'),fullfile(voc_base_dir,'ImageSets','Segmentation','val.txt'),fullfile('..','training','data_lists','val_labels.txt'),'.png');

if create_edge_gt
    fprintf('Creating edge ground truth for training images \n');
    create_edge_ground_truth(fullfile('..','training','data_lists','train_labels.txt'),fullfile(voc_base_dir,'EdgeGt'));
    fprintf('Creating edge ground truth for validation images \n');
    create_edge_ground_truth(fullfile('..','training','data_lists','val_labels.txt'),fullfile(voc_base_dir,'EdgeGt'));
    
    fprintf('Writing list files for edges\n')
    write_list(fullfile(voc_base_dir,'EdgeGt'),fullfile(voc_base_dir,'ImageSets','Segmentation','train.txt'),fullfile('..','training','data_lists','train_edge_x.txt'),'_x.png');
    write_list(fullfile(voc_base_dir,'EdgeGt'),fullfile(voc_base_dir,'ImageSets','Segmentation','val.txt'),fullfile('..','training','data_lists','val_edge_x.txt'),'_x.png');
    write_list(fullfile(voc_base_dir,'EdgeGt'),fullfile(voc_base_dir,'ImageSets','Segmentation','train.txt'),fullfile('..','training','data_lists','train_edge_y.txt'),'_y.png');
    write_list(fullfile(voc_base_dir,'EdgeGt'),fullfile(voc_base_dir,'ImageSets','Segmentation','val.txt'),fullfile('..','training','data_lists','val_edge_y.txt'),'_y.png');
    write_list(fullfile(voc_base_dir,'EdgeGt'),fullfile(voc_base_dir,'ImageSets','Segmentation','train.txt'),fullfile('..','training','data_lists','train_edge_no.txt'),'_no.png');
    write_list(fullfile(voc_base_dir,'EdgeGt'),fullfile(voc_base_dir,'ImageSets','Segmentation','val.txt'),fullfile('..','training','data_lists','val_edge_no.txt'),'_no.png');
end

end

function save_as_raw_png(im_dir, orig_list, res_dir)
if ~exist(res_dir,'dir')
    mkdir(res_dir)
end
f_in = fopen(orig_list);
tline = fgetl(f_in);
while ischar(tline)
    [im, map] = imread(sprintf('%s%s',fullfile(im_dir,tline),'.png'));
    imwrite(im,fullfile(res_dir,[tline '.png']));
    tline = fgetl(f_in);
end
fclose(f_in);
end

function write_list(im_dir, orig_list, new_list, ext)
f_in = fopen(orig_list);
f_out = fopen(new_list,'w');

tline = fgetl(f_in);
while ischar(tline)
    fprintf(f_out,'%s%s\n',fullfile(im_dir,tline),ext);
    tline = fgetl(f_in);
end
fclose(f_out);
fclose(f_in);
end

function create_edge_ground_truth(list_in,folder_out,useallpixels,n_images_to_plot,smoothedges)
if nargin < 5
    smoothedges = true;
end
if nargin < 4
    n_images_to_plot = 0;
end
if nargin < 3
    useallpixels = false;
end

if ~exist(folder_out,'dir')
    mkdir(folder_out)
end

f_in = fopen(list_in);
tline = fgetl(f_in);
while ischar(tline)
    gt = imread(tline);
    [edgemapx,edgemapy,edge_no] = find_edges(gt,useallpixels,n_images_to_plot,smoothedges);
    
    s_ind = strfind(tline,'/');
    f_name = tline(s_ind(end)+1:end-4);
    imwrite(edgemapx,fullfile(folder_out,[f_name '_x.png']));
    imwrite(edgemapy,fullfile(folder_out,[f_name '_y.png']));
    imwrite(edge_no,fullfile(folder_out,[f_name '_no.png']));
    tline = fgetl(f_in);
end
end

function [edgeX,edgeY,edge_no] = find_edges(seg,useallpixels,n_images_to_plot,smoothedges)
Gx_tot = zeros(size(seg));
Gy_tot = zeros(size(seg));

classes = unique(seg(:))';
for c = classes
    if c == 255
        continue;
    end
    
    thisclass = seg == c;
    [Gx, Gy] = imgradientxy(thisclass);
    indstochange = Gx.^2+Gy.^2 > Gx_tot.^2+Gy_tot.^2;
    
    Gx_tot(indstochange) = Gx(indstochange);
    Gy_tot(indstochange) = Gy(indstochange);
end

if smoothedges
    x_filt = 1/0.1370*fspecial('gaussian', [1 13], 3);
    y_filt = 1/0.1370*fspecial('gaussian', [13 1], 3);
    Gx_tot = imfilter(Gx_tot,x_filt,'symmetric'); 
    Gy_tot = imfilter(Gy_tot,y_filt,'symmetric');
end

Gx_tot = abs(Gx_tot)./max(abs(Gx_tot(:)));
Gy_tot = abs(Gy_tot)./max(abs(Gy_tot(:)));
no_edge = -sqrt(Gx_tot.^2 + Gy_tot.^2);
no_edge = (no_edge-min(no_edge(:)))/(max(no_edge(:))-min(no_edge(:)));

Gx_tot = floor(200*Gx_tot+25);
Gy_tot = floor(200*Gy_tot+25);
no_edge = floor(200*no_edge+25);

if useallpixels
    ignore_inds = seg == 255 & Gx_tot == 0 & Gx_tot == 0;
else
    % set all pixels not close to an edge to ignore inds
    tmp = no_edge < 0.9 * max(no_edge(:));
    tmp = imdilate(tmp,ones(30));
    ignore_inds = ~tmp | (seg == 255 & Gx_tot == 0 & Gx_tot == 0);
end

Gx_tot(ignore_inds) = 255;
Gy_tot(ignore_inds) = 255;
no_edge(ignore_inds) = 255;
if n_images_to_plot > 0
    subplot(2,3,3)
    imshow(ignore_inds,[])
    title('pixels2ignore')
    drawnow
end
edgeX = uint8(Gx_tot);
edgeY = uint8(Gy_tot);
edge_no = uint8(no_edge);

end