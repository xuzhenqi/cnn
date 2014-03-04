function theta = cnnInitParams(cnnConfig)
% Initialize parameters
%                            
% Parameters:
%  cnnConfig    - cnn configuration variable
%
% Returns:
%  theta      -  parameter structure

numLayers = size(cnnConfig.layer,2);
for i = 1 : numLayers
    tempLayer = cnnConfig.layer{i};
    switch tempLayer.type
        case 'input'
            theta.W{i} = [];
            theta.b{i} = [];
            row = tempLayer.dimension(1);
            col = tempLayer.dimension(2);
            channel = tempLayer.dimension(3);
        case 'conv'        
            row = row + 1 - tempLayer.filterDim(1);
            col = col + 1 - tempLayer.filterDim(2);          
            theta.W{i} = 1e-1*randn(row,col,channel,tempLayer.numFilters);
            channel = tempLayer.numFilters;
            theta.b{i} = zeros(channel, 1);
        case 'pool'
            theta.W{i} = [];
            theta.b{i} = [];
            row = int32(row/tempLayer.poolDim(1));
            col = int32(col/tempLayer.poolDim(2));
        case 'stack2line'
            theta.W{i} = [];
            theta.b{i} = [];
            row = row * col * channel;
            col = 1;
            channel = 1;
            dimension = row;
        case {'sigmoid','tanh','relu','softmax','softsign'}
            % initialisation of dnn method
            r = sqrt(6) ./ sqrt(double(dimension) + tempLayer.dimension);
            theta.W{i} = rand(tempLayer.dimension, dimension) * 2 .* r - r;
            dimension = tempLayer.dimension;
            theta.b{i} = zeros(dimension, 1);
    end
end
end

