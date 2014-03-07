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
%  labels     -  for softmax output layer and cross entropy cost function,
%  the labels are the class numbers.
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
numImages = size(images,4);

temp = cell(numLayers, 1);
grad = cell(numLayers, 1);
temp{1}.after = images;

for l = 2 : numLayers
    tempLayer = cnnConfig.layer{l};
    tempTheta = theta{l};
    switch tempLayer.type
        case 'conv'
            temp{l}.after = cnnConvolve(temp{l-1}.after, tempTheta.W, tempTheta.b, tempLayer.nonLinaerType, tempLayer.conMatrix); 
        case 'pool'
            [temp{l}.after, temp{l}.weights] = cnnPool(tempLayer.poolDim, temp{l-1}.after, tempLayer.poolType);
        case 'stack2line'
            temp{l}.after = reshape(temp{l-1}.after, [], numImages);
        case {'sigmoid','tanh','relu','softmax','softsign'}
            temp{l}.after = nonlinear(temp{l-1}.after, tempTheta.W, tempTheta.b, tempLayer.type);
    end
end

%%======================================================================
%% STEP 1b: Calculate Cost
% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(temp{numLayers}.after,[],1);
    preds = preds';
    return;
end;

switch cnnConfig.costFun
    case 'crossEntropy'
        numClasses = cnnConfig.layer{numLayers}.dimension;
        extLabels = zeros(numClasses, numImages);
        extLabels(sub2ind(size(extLabels), labels', 1 : numImages)) = 1;
        cost = - sum(sum(extLabels .* log(temp{numLayers}.after))) / numImages;
end

%%======================================================================
%% STEP 1c: Backpropagation
if strcmp(cnnConfig.costFun, 'crossEntropy') && strcmp(tempLayer.type, 'softmax')
    temp{l}.gradBefore = temp{l}.after - extLabels;
    grad{l}.W = temp{l}.gradBefore * temp{l - 1}.after' / numImages;
    grad{l}.b = mean(temp{l}.gradBefore, 2);
end

for l = numLayers-1 : -1 : 2
    tempLayer = cnnConfig.layer{l};
    switch tempLayer.type
        case 'sigmoid'
            temp{l}.gradBefore = theta{l + 1}.W * temp{l + 1}.gradBefore .* temp{l}.after .* (1 - temp{l}.after); 
            grad{l}.W = temp{l}.gradBefore * temp{l - 1}.after' / numImages;
            grad{l}.b = mean(temp{l}.gradBefore, 2);
        case 'stack2line'
            temp{l}.gradBefore = reshape(theta{l + 1}.W * temp{l + 1}.gradBefore, size(temp{l - 1}.after));
            break;
    end
end

for l = l - 1 : -1 : 2
    tempLayer = cnnConfig.layer{l};
    switch tempLayer.type 
        case 'pool'
            numChannel = size(temp{l}.after);
            temp{l}.gradAfter = zeros(size(temp{l}.after));
            if isempty(theta{l + 1}.W)
                % the upper layer is 'stack2line'
                temp{l}.gradAfter = temp{l + 1}.gradBefore;
            else
                % the upper layer is 'conv'
                for i = 1 : numImages
                    for c = 1 : numChannel
                        for j = 1 : size(temp{l + 1}.gradBefore, 3)
                            if cnnConfig.layer{l + 1}.conMatrix(c,j) ~= 0
                                temp{l}.gradAfter(:,:,c,i) = temp{l}.gradAfter(:,:,c,i) + conv2(temp{l + 1}.gradBefore(:,:,j,i), rot90(theta{l + 1}.W(:,:,c,j)), 'full');
                            end
                        end
                    end
                end
            end
            temp{l}.gradBefore = size(temp{l - 1}.after);
            for i = 1 : numImages
                for c = 1 : numChannel
                    temp{l}.gradBefore(:,:,c,i) = kron(temp{l}.gradAfter(:,:,c,i), ones(tempLayer.poolDim)) .* temp{l}.weights;
                end
            end
        case 'conv'
            switch tempLayer.nonLinearType
                case 'sigmoid'
                    temp{l}.gradBefore = temp{l + 1}.gradBefore .* temp{l}.after .* (1 - temp{l}.after);
            end
            tempW = zeors([size(theta{l}.W) numImages]); 
            numInputMap = size(grad{l}.W, 3);
            numOutputMap = size(grad{l}.W, 4);
            for i = 1 : numImages
                for nI = 1 : numInputMap
                    for nO = 1 : numOutputMap
                        if tempLayer.conMatrix(nI,nO) ~= 0
                            temp.W(:,:,nI,nO,i) = conv2(temp{l - 1}.after(:,:,nI,i), rot90(temp{l}.gradBefore(:,:,nO,i), 2), 'valid');
                        end
                    end
                end
            end
            grad{l}.W = mean(tempW,5);
            grad{l}.b = mean(sum(sum(temp{l}.gradBefore)),2);                   
        otherwise 
            printf('%s layer is not supported', tempLayer.type);       
    end
end

%% Unroll gradient into grad vector for minFunc

end
