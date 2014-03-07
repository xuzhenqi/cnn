function [theta, meta] = cnnInitParams(cnnConfig)
% Initialize parameters
%                            
% Parameters:
%  cnnConfig    - cnn configuration variable
%
% Returns:
%  theta      -  parameter structure
%  meta       -  meta param 
%       numTotalParams : total number of the parameters
%       numParams      : the number of the parameters each layer

numLayers = size(cnnConfig.layer,2);
theta = cell(numLayers,1);
numParams = zeros(numLayers,1);
for i = 1 : numLayers
    tempLayer = cnnConfig.layer{i};
    switch tempLayer.type
        case 'input'
            theta{i}.W = [];
            theta{i}.b = [];
            row = tempLayer.dimension(1);
            col = tempLayer.dimension(2);
            channel = tempLayer.dimension(3);
        case 'conv'        
            row = row + 1 - tempLayer.filterDim(1);
            col = col + 1 - tempLayer.filterDim(2);          
            theta{i}.W = 1e-1*randn(row,col,channel,tempLayer.numFilters);
            numParams(i) = row * col * channel * tempLayer.numFilters + tempLayer.numFilters;
            channel = tempLayer.numFilters;
            theta{i}.b = zeros(channel, 1);
        case 'pool'
            theta{i}.W = [];
            theta{i}.b = [];
            row = int32(row/tempLayer.poolDim(1));
            col = int32(col/tempLayer.poolDim(2));
        case 'stack2line'
            theta{i}.W = [];
            theta{i}.b = [];
            row = row * col * channel;
            col = 1;
            channel = 1;
            dimension = row;
        case {'sigmoid','tanh','relu','softmax','softsign'}
            % initialisation of dnn method
            r = sqrt(6) ./ sqrt(double(dimension) + tempLayer.dimension);
            theta{i}.W = rand(tempLayer.dimension, dimension) * 2 .* r - r;
            numParams(i) = tempLayer.dimension * (dimension + 1);
            dimension = tempLayer.dimension;
            theta{i}.b = zeros(dimension, 1);
    end
end
meta.numTotalParams = sum(numParams);
meta.numParams = numParams;
meta.numLayers = numLayers;
end
