function pooledFeatures = cnnPool(poolDim, convolvedFeatures, pooltypes)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region, is a 1 * 2 vector(poolDimRow poolDimCol);
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

if nargin < 3
    pooltypes = 'meanpool';
end

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDimRow = size(convolvedFeatures, 1);
convolvedDimCol = size(convolvedFeatures, 2);

pooledFeatures = zeros(int32(convolvedDimRow / poolDim(1)), ...
        int32(convolvedDimCol / poolDim(2)), numFilters, numImages);

poolFilter = ones(poolDim) * 1/poolDim(1)/poolDim(2);

for imageNum = 1:numImages
    for filterNum = 1:numFilters
	features = convolvedFeatures(1:size(pooledFeatures,1)*poolDim(1),1:size(pooledFeatures,2)*poolDim(2),filterNum,imageNum);
    switch pooltypes
        case 'meanpool'
            poolConvolvedFeatures = conv2(features,poolFilter,'valid');
            pooledFeatures(:,:,filterNum,imageNum) = poolConvolvedFeatures(1:poolDim:end,1:poolDim:end);
        case 'maxpool'
            % Todo : this is need to be optimized.
            pooledFeatures(:,:,filterNum,imageNum) = blockproc(features,poolDim,@maxblock);
        otherwise
            error(message('cnn:cnnPool:WrongLayertypes'));
    end
    end
end

end
 
function b = maxblock(a)
    b = max(max(a.data));
end

