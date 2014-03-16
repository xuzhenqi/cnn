%% Convolution Neural Network Exercise

% diary on;
% diary('run_log');

%% STEP 0: Initialize Parameters and Load Data
%  complete the config.m to config the network structure;
cnnConfig = config();
%  calling cnnInitParams() to initialize parameters
[theta meta] = cnnInitParams(cnnConfig);

% Load MNIST Data
images = loadMNISTImages('train-images-idx3-ubyte');
d = cnnConfig.layer{1}.dimension;
images = reshape(images,d(1),d(2),d(3),[]);
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 3;
options.minibatch = 128;
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta),theta,images,labels,options);

%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

testImages = loadMNISTImages('t10k-images-idx3-ubyte');
testImages = reshape(testImages,d(1),d(2),d(3),[]);
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

[cost,grad,preds]=cnnCost(opttheta,testImages,testLabels,cnnConfig,meta,true);

acc = sum(preds==testLabels)/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);

%diary off;
