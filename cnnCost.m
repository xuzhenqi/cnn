function [cost, grad, preds] = cnnCost(theta, images, labels,cnnConfig, meta, pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  a vector parameter
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



theta = thetaChange(theta,meta,'vec2stack',cnnConfig);
%%======================================================================
%% STEP 1a: Forward Propagation
numLayers = size(theta, 1);
numImages = size(images,4);
layersizes = meta.layersize;



temp = cell(numLayers, 1);
grad = cell(numLayers, 1);
temp{1}.after = images;
assert(isequal(size(images),[layersizes{1} numImages]),'layersize do not match at layer 1');

for l = 2 : numLayers
    tempLayer = cnnConfig.layer{l};
    tempTheta = theta{l};
    switch tempLayer.type
        case 'conv'
            [temp{l}.after, temp{l}.linTrans] = cnnConvolve(temp{l-1}.after, tempTheta.W, tempTheta.b, tempLayer.nonLinearType, tempLayer.conMatrix);         
        case 'pool'
            [temp{l}.after, temp{l}.weights] = cnnPool(tempLayer.poolDim, temp{l-1}.after, tempLayer.poolType);
        case 'stack2line'
            temp{l}.after = reshape(temp{l-1}.after, [], numImages);
        case {'sigmoid','tanh','relu','softmax'}
            temp{l}.after = nonlinear(temp{l-1}.after, tempTheta.W, tempTheta.b, tempLayer.type);
        case 'softsign'
            [temp{l}.after, temp{l}.linTrans] = nonlinear(temp{l-1}.after, tempTheta.W, tempTheta.b, tempLayer.type);
    end
    assert(isequal(size(temp{l}.after),[layersizes{l} numImages]),'layersize do not match at layer %d\n',l);
end

%%======================================================================
%% STEP 1b: Calculate Cost
% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(temp{numLayers}.after,[],1);
    preds = preds';
    cost = 0;
    grad = 0;
    return;
end;

switch cnnConfig.costFun
    case 'crossEntropy'
        numClasses = cnnConfig.layer{numLayers}.dimension;
        extLabels = zeros(numClasses, numImages);
        extLabels(sub2ind(size(extLabels), labels', 1 : numImages)) = 1;
        cost = - mean(sum(extLabels .* log(temp{numLayers}.after)));
end

%%======================================================================
%% STEP 1c: Backpropagation
if strcmp(cnnConfig.costFun, 'crossEntropy') && strcmp(tempLayer.type, 'softmax')
    temp{l}.gradBefore = temp{l}.after - extLabels;
    grad{l}.W = temp{l}.gradBefore * temp{l - 1}.after' / numImages;
    grad{l}.b = mean(temp{l}.gradBefore, 2);
end
assert(isequal(size(grad{l}.W),size(theta{l}.W)),'size of layer %d .W do not match',l);
assert(isequal(size(grad{l}.b),size(theta{l}.b)),'size of layer %d .b do not match',l);

for l = numLayers-1 : -1 : 2
    tempLayer = cnnConfig.layer{l};
    switch tempLayer.type
        case 'sigmoid'
            temp{l}.gradBefore = theta{l + 1}.W' * temp{l + 1}.gradBefore .* temp{l}.after .* (1 - temp{l}.after); 
            grad{l}.W = temp{l}.gradBefore * temp{l - 1}.after' / numImages;
            grad{l}.b = mean(temp{l}.gradBefore, 2);
            
        case 'relu'
            temp{l}.gradBefore = theta{l + 1}.W' * temp{l + 1}.gradBefore .* (temp{l}.after > 0);
            grad{l}.W = temp{l}.gradBefore * temp{l - 1}.after' / numImages;
            grad{l}.b = mean(temp{l}.gradBefore, 2);
        
        case 'tanh'
            temp{l}.gradBefore = theta{l + 1}.W' * temp{l + 1}.gradBefore .* (1 - temp{l}.after .^ 2);
            grad{l}.W = temp{l}.gradBefore * temp{l - 1}.after' / numImages;
            grad{l}.b = mean(temp{l}.gradBefore, 2);
            
        case 'softsign'
            temp{l}.gradBefore = theta{l + 1}.W' * temp{l + 1}.gradBefore ./ ((1 + abs(temp{l}.linTrans)) .^ 2);
            grad{l}.W = temp{l}.gradBefore * temp{l - 1}.after' / numImages;
            grad{l}.b = mean(temp{l}.gradBefore, 2);
            
        case 'stack2line'
            temp{l}.gradBefore = reshape(theta{l + 1}.W' * temp{l + 1}.gradBefore, size(temp{l - 1}.after));
            grad{l}.W = [];
            grad{l}.b = [];
            break;
    end
    assert(isequal(size(grad{l}.W),size(theta{l}.W)),'size of layer %d .W do not match',l);
    assert(isequal(size(grad{l}.b),size(theta{l}.b)),'size of layer %d .b do not match',l);
end

for l = l - 1 : -1 : 2
    tempLayer = cnnConfig.layer{l};
    switch tempLayer.type 
        case 'pool'
            numChannel = size(temp{l}.after,3);
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
                                temp{l}.gradAfter(:,:,c,i) = temp{l}.gradAfter(:,:,c,i) + conv2(temp{l + 1}.gradBefore(:,:,j,i), theta{l + 1}.W(:,:,c,j), 'full');
                            end
                        end
                    end
                end
            end
            
            % Todo
            % If there are cropped borders, situations may be more
            % complicated.
            
            temp{l}.gradBefore = zeros(size(temp{l - 1}.after));
            for i = 1 : numImages
                for c = 1 : numChannel
                    temp{l}.gradBefore(:,:,c,i) = kron(temp{l}.gradAfter(:,:,c,i), ones(tempLayer.poolDim)) .* temp{l}.weights(:,:,c,i);
                end
            end
            grad{l}.W = [];
            grad{l}.b = [];
        case 'conv'
            switch tempLayer.nonLinearType
                case 'sigmoid'
                    temp{l}.gradBefore = temp{l + 1}.gradBefore .* temp{l}.after .* (1 - temp{l}.after);
                case 'tanh'
                    temp{l}.gradBefore = temp{l + 1}.gradBefore .* (1 - temp{l}.after .^ 2);
                case 'softsign'
                    temp{l}.gradBefore = temp{l + 1}.gradBefore ./ ((1 + abs(temp{l}.linTrans)) .^ 2);
                case 'relu'
                    temp{l}.gradBefore = temp{l + 1}.gradBefore .* (temp{l}.after > 0);
            end
            tempW = zeros([size(theta{l}.W) numImages]); 
            numInputMap = size(tempW, 3);
            numOutputMap = size(tempW, 4);
            for i = 1 : numImages
                for nI = 1 : numInputMap
                    for nO = 1 : numOutputMap
                        if tempLayer.conMatrix(nI,nO) ~= 0
                            tempW(:,:,nI,nO,i) = conv2(temp{l - 1}.after(:,:,nI,i), rot90(temp{l}.gradBefore(:,:,nO,i), 2), 'valid');
                        end
                    end
                end
            end
            grad{l}.W = mean(tempW,5);
            tempb = mean(sum(sum(temp{l}.gradBefore)),4); 
            grad{l}.b = tempb(:);
                             
        otherwise 
            printf('%s layer is not supported', tempLayer.type);       
    end
    assert(isequal(size(grad{l}.W),size(theta{l}.W)),'size of layer %d .W do not match',l);
    assert(isequal(size(grad{l}.b),size(theta{l}.b)),'size of layer %d .b do not match',l); 
end

%% Unroll gradient into grad vector for minFunc
grad = thetaChange(grad,meta,'stack2vec',cnnConfig);

end
