function cnnConfig = configTestGradient()
cnnConfig.layer{1}.type = 'input';
cnnConfig.layer{1}.dimension = [28 28 1];

% cnnConfig.layer{2}.type = 'conv';
% cnnConfig.layer{2}.filterDim = [9 9];
% cnnConfig.layer{2}.numFilters = 3;
% cnnConfig.layer{2}.nonLinearType = 'sigmoid';
% cnnConfig.layer{2}.conMatrix = ones(cnnConfig.layer{1}.dimension(3),cnnConfig.layer{2}.numFilters);

% cnnConfig.layer{3}.type = 'pool';
% cnnConfig.layer{3}.poolDim = [2 2];
% cnnConfig.layer{3}.poolType = 'meanpool';

cnnConfig.layer{2}.type = 'stack2line';

% cnnConfig.layer{5}.type = 'sigmoid';
% cnnConfig.layer{5}.dimension = 60;

cnnConfig.layer{3}.type = 'softmax';
cnnConfig.layer{3}.dimension = 10;

cnnConfig.costFun = 'crossEntropy';
end