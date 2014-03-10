%% Convolution Neural Network Exercise

% diary on;
% diary('run_log');

%% STEP 0: Initialize Parameters and Load Data
%  complete the config.m to config the network structure;
cnnConfig = config();
%  calling cnnInitParams() to initialize parameters
[theta meta] = cnnInitParams(cnnConfig);

% Load MNIST Data
images = loadMNISTImages('../Dataset/MNIST/train-images-idx3-ubyte');
images = reshape(images,cnnConfig.layer{1}.dimension,[]);
labels = loadMNISTLabels('../Dataset/MNIST/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 50;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim,...
                      numFilters,poolDim),theta,images,labels,options);

%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

testImages = loadMNISTImages('../common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('../common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                filterDim,numFilters,poolDim,true);

acc = sum(preds==testLabels)/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);

diary off;
