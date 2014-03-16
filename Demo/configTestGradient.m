function cnnConfig = configTestGradient()

l = 1;

cnnConfig.layer{l}.type = 'input';
cnnConfig.layer{l}.dimension = [28 28 1];
l = l + 1;

cnnConfig.layer{l}.type = 'conv';
cnnConfig.layer{l}.filterDim = [9 9];
cnnConfig.layer{l}.numFilters = 3;
cnnConfig.layer{l}.nonLinearType = 'softsign';
cnnConfig.layer{l}.conMatrix = ones(1,3);
l = l + 1;

cnnConfig.layer{l}.type = 'pool';
cnnConfig.layer{l}.poolDim = [2 2];
cnnConfig.layer{l}.poolType = 'meanpool';
l = l + 1;

cnnConfig.layer{l}.type = 'conv';
cnnConfig.layer{l}.filterDim = [5 5];
cnnConfig.layer{l}.numFilters = 4;
cnnConfig.layer{l}.nonLinearType = 'sigmoid';
cnnConfig.layer{l}.conMatrix = ones(3,4);
l = l + 1;

cnnConfig.layer{l}.type = 'pool';
cnnConfig.layer{l}.poolDim = [2 2];
cnnConfig.layer{l}.poolType = 'maxpool';
l = l + 1;

cnnConfig.layer{l}.type = 'stack2line';
l = l + 1;

% cnnConfig.layer{l}.type = 'tanh';
% cnnConfig.layer{l}.dimension = 60;
% l = l + 1;

cnnConfig.layer{l}.type = 'softmax';
cnnConfig.layer{l}.dimension = 10;
l = l + 1;

cnnConfig.costFun = 'crossEntropy';
end