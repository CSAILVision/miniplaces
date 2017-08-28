% this baseline model is implemented in matconvnet, 
% please install MatConvNet first at http://www.vlfeat.org/matconvnet/

% load pre-trained model

load('categoryIDX.mat');
path_model = 'refNet1-epoch-60.mat';
if ~exist(path_model)
    system(['wget miniplaces.csail.mit.edu/model/' path_model])
end
load([path_model]) ;

% load and preprocess an image
im = imread('img2.jpg') ;
im_resize = imresize(im, net.normalization.imageSize(1:2)) ;
im_ = single(im_resize) ; 
for i=1:3
    im_(:,:,i) = im_(:,:,i)-net.normalization.averageImage(i);
end

% change the last layer of CNN from softmaxloss to softmax
net.layers{1,end}.type = 'softmax';
net.layers{1,end}.name = 'prob';

% run the CNN
res = vl_simplenn(net, im_) ;

scores = squeeze(gather(res(end).x)) ;
[score_sort, idx_sort] = sort(scores,'descend') ;
figure, imagesc(im_resize) ;
for i=1:5
    disp(sprintf('%s (%d), score %.3f', categoryIDX{idx_sort(i),1}, idx_sort(i), score_sort(i)));
end
