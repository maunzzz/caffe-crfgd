% if not working try starting matlab using
% "LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21" matlab"

%% SETUP
data_set = 'pascal'; % 'pascal', add more options
data_root = '/media/cvia/disk2/Data/test_pascal/VOCdevkit/VOC2012'; %point this to pascal root folder
imname = '2007_002227';

model = '../training/filterbank/net_pretrain_deploy.prototxt';
weights = '../training/snapshots/filterbank_pretrain_iter_92232.caffemodel'; 

%% Do stuff
caffe_matlab_root = '../../matlab';
addpath(caffe_matlab_root);
caffe.reset_all();

switch data_set
    case 'pascal'
        im_path = fullfile(data_root,'JPEGImages',[imname '.jpg']);
        seg_path = fullfile(data_root,'SegmentationClass',[imname '.png']);
        edge_path1 = fullfile(data_root,'EdgeGt',[imname '_x.png']);
        edge_path2 = fullfile(data_root,'EdgeGt',[imname '_y.png']);
        edge_path3 = fullfile(data_root,'EdgeGt',[imname '_no.png']);
end

gt = imread(seg_path);
edge1 = imread(edge_path1);
edge2 = imread(edge_path2);
edge3 = imread(edge_path3);
im = imread(im_path);

switch data_set
    case 'pascal'
        im_mean = zeros(1,1,3);
        im_mean(1,1,1) = 104.008;
        im_mean(1,1,2) = 116.669;
        im_mean(1,1,3) = 122.675;
        ims = [640 640];
end

caffe_input = zeros(ims(1),ims(2),3,1,'single');
caffe_input(1:size(im,1),1:size(im,2),:,1) = single(bsxfun(@minus,double(im(:,:,[3 2 1])),im_mean));
caffe_input = permute(caffe_input,[2 1 3 4]);
caffe.set_mode_cpu();

net = caffe.Net(model, weights, 'test');    

net.blobs('data').reshape([ims(2) ims(1) 3 1]);
net.reshape();
res = net.forward({caffe_input});

edges = res{1};                                    
edges = permute(edges,[2 1 3]);

probs = res{2};                                    
probs = permute(probs,[2 1 3]);
 
[~,seg]= max(probs(1:size(im,1),1:size(im,2),:),[],3);           
seg(1,1) = 0;
 
gt = gt + 1;
gt(gt == 255) = 0;
 
e1 = edge1 == 255;
e2 = edge2 == 255;
e3 = edge3 == 255;
edge1 = edge1 + 1; 
edge2 = edge2 + 1;
edge3 = edge3 + 1;
edge1(e1) = 0;
edge2(e2) = 0;
edge3(e3) = 0;

figure;
subplot(3,3,1);imshow(im);title('im');
subplot(3,3,2);imshow(seg,[]);title('seg');                                                                                                                                                                           
subplot(3,3,3);imshow(gt,[]);title('gt');  
subplot(3,3,4);imshow(256*edges(1:size(im,1),1:size(im,2),1),[]);title('edge-x');ylabel('net output')     
subplot(3,3,5);imshow(256*edges(1:size(im,1),1:size(im,2),2),[]);title('edge-y');     
subplot(3,3,6);imshow(256*edges(1:size(im,1),1:size(im,2),3),[]);title('edge-no');     
subplot(3,3,7);imshow(edge1,[]);ylabel('gt')     
subplot(3,3,8);imshow(edge2,[]);
subplot(3,3,9);imshow(edge3,[]);