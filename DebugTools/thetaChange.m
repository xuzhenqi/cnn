function newTheta = thetaChange(oldTheta, meta, type, cnnConfig)
%thetaChange change theta from the old form to the new form based on type
% type  - change type
%       'stack2vec'
%       'vec2stack'

if exist('type', 'var')
    type = 'stack2vec';
end

switch type
    case 'stack2vec'
        newTheta = zeros(meta.numTotalParams, 1);
        cur = 1;
        for i = 1 : meta.numLayers
            if meta.numParams(i) ~= 0
                len = length(oldTheta{i}.W(:));
                newTheta(cur:cur+len-1) = oldTheta{i}.W(:);
                cur = cur + len;
                len = length(oldTheta{i}.b(:));
                newTheta(cur:cur+len-1) = oldTheta{i}.b(:);
                cur = cur + len;
            end
        end
    case 'vec2stack'
        newTheta = cell(size(cnnConfig));
        for i = 1 : size(newTheta)
            tempLayer = cnnConfig.layer{i};
            switch tempLayer.type
                case 'input'
                    newTheta{i}.W = [];
                    newTheta{i}.b = [];
                    row = tempLayer.dimension(1);
                    col = tempLayer.dimension(2);
                    channel = tempLayer.dimension(3);
                case 'conv'        
                    row = row + 1 - tempLayer.filterDim(1);
                    col = col + 1 - tempLayer.filterDim(2);          
                    newTheta{i}.W = reshape(oldTheta(1:row*col*channel*tempLayer.numFilters),row,col,channel,tempLayer.numFilters);
                    oldTheta(1:row*col*channel*tempLayer.numFilters)=[];
                    channel = tempLayer.numFilters;
                    newTheta{i}.b = oldTheta(1:channel);
                case 'pool'
                    newTheta{i}.W = [];
                    newTheta{i}.b = [];
                    row = int32(row/tempLayer.poolDim(1));
                    col = int32(col/tempLayer.poolDim(2));
                case 'stack2line'
                    newTheta{i}.W = [];
                    newTheta{i}.b = [];
                    row = row * col * channel;
                    col = 1;
                    channel = 1;
                    dimension = row;
                case {'sigmoid','tanh','relu','softmax','softsign'}
                    % initialisation of dnn method
                    newTheta{i}.W = reshape(oldTheta(1:tempLayer.dimension*dimension),tempLayer.dimension, dimension);
                    oldTheta(1:tempLayer.dimension*dimension) = [];
                    dimension = tempLayer.dimension;
                    newTheta{i}.b = oldTheta(1:dimension);
                    oldTheta(1:dimension) = [];
            end
        end
end


end