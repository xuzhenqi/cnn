function newTheta = thetaChange(oldTheta, meta, type, cnnConfig)
%thetaChange change theta from the old form to the new form based on type
% type  - change type
%       'stack2vec'
%       'vec2stack'

if ~exist('type', 'var')
    type = 'stack2vec';
end

switch type
    case 'stack2vec'
        newTheta = zeros(meta.numTotalParams, 1);
        cur = 1;
        for i = 1 : meta.numLayers
            if meta.numParams(i,1) ~= 0
                newTheta(cur:cur+meta.numParams(i,1)-1) = oldTheta{i}.W(:);
                cur = cur + meta.numParams(i,1);
                newTheta(cur:cur+meta.numParams(i,2)-1) = oldTheta{i}.b(:);
                cur = cur + meta.numParams(i,2);
            end
        end
    case 'vec2stack'
        newTheta = cell(meta.numLayers,1);
        for i = 1 : meta.numLayers
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
                    newTheta{i}.W = reshape(oldTheta(1:meta.numParams(i,1)),meta.paramsize{i});
                    oldTheta(1:meta.numParams(i,1))=[];
                    channel = tempLayer.numFilters;
                    newTheta{i}.b = oldTheta(1:channel);
                    oldTheta(1:channel) = [];
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
                    %dimension = row;
                case {'sigmoid','tanh','relu','softmax','softsign'}
                    % initialisation of dnn method
                    newTheta{i}.W = reshape(oldTheta(1:meta.numParams(i,1)),meta.paramsize{i});
                    oldTheta(1:meta.numParams(i,1)) = [];
                    dimension = tempLayer.dimension;
                    newTheta{i}.b = oldTheta(1:dimension);
                    oldTheta(1:dimension) = [];
            end
        end
        assert(isempty(oldTheta), 'Error: oldTheta is not empty!\n');
end
end