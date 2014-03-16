function cnnConfig = config()
cnnConfig.layer{1}.type = 'input';
cnnConfig.layer{1}.dimension = [28 28 1];

cnnConfig.layer{2}.type = 'conv';
cnnConfig.layer{2}.filterDim = [9 9];
cnnConfig.layer{2}.numFilters = 20;
cnnConfig.layer{2}.nonLinearType = 'sigmoid';
cnnConfig.layer{2}.conMatrix = ones(1,20);

cnnConfig.layer{3}.type = 'pool';
cnnConfig.layer{3}.poolDim = [2 2];
cnnConfig.layer{3}.poolType = 'meanpool';

cnnConfig.layer{4}.type = 'stack2line';

cnnConfig.layer{5}.type = 'sigmoid';
cnnConfig.layer{5}.dimension = 360;

cnnConfig.layer{6}.type = 'sigmoid';
cnnConfig.layer{6}.dimension = 60;

cnnConfig.layer{7}.type = 'softmax';
cnnConfig.layer{7}.dimension = 10;

cnnConfig.costFun = 'crossEntropy';
end