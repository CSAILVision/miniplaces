% Here we provide the reference network (refNet1) trained on mini-Places.
% you could check out run_miniplacesCNN.m to see how the network is loaded
% and used to predict the scene category of some image. Please intall the
% MatConvnet properly first, following this instruction 
% http://www.vlfeat.org/matconvnet/quick/

% refNet1 has top1 accuracy as 0.355 and top5 accuracy as 0.649 on the
% validation set of mini-Places.

% We also provide the sample code sample_refNet_initial.m to show how the
% refNet1 is initialized before training. You could adapt it to your model
% training.

% I highly recommend you to go through the two examples of training model on mnist
% and cifar included in the MatConvnet before you train your own network on
% miniplaces.

% Bolei Zhou, Sep.30, 2015.