function [cost, grad, preds] = cnnCost(cnnConfig, theta, images, labels, pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  cnnConfig  -  cnn configuration
%  theta      -  parameter structure
%  images     -  stores images in imageDim x imageDim x channel x numImges
%                array    
%  labels     - 
%  pred       -  boolean only forward propagate and return
%                predictions
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)

if ~exist('pred','var')
    pred = false;
end;

%%======================================================================
%% STEP 1a: Forward Propagation
numLayers = size(theta.W, 2);
temp = cell(numLayers, 1);
temp{1}.after = images;

for l = 2 : numLayers
    tempLayer = cnnConfig.layer{l};
    tempTheta = theta{l};
    switch tempLayer.type
        case 'conv'
            temp{l}.after = cnnConvolve(temp{l-1}.after, tempTheta.W, tempTheta.b, tempLayer.nonLinaerType, tempLayer.conMatrix); 
        case 'pool'
            temp{l}.after = cnnPool(tempLayer.poolDim, temp{l-1}.after, tempLayer.poolType);
        case 'stack2line'
            temp{l}.after = temp{l-1}.after(:);
        case {'sigmoid','tanh','relu','softmax','softsign'}
            temp{l}.after = nonlinear(temp{l-1}.after, tempTheta.W, tempTheta.b, tempLayer.type);
    end
end

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.


% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.


%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%

%% Unroll gradient into grad vector for minFunc

end
