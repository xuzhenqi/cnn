function cnnConfig = configTestGradient()
cnnConfig.layer{1}.type = 'input';
cnnConfig.layer{1}.dimension = [28 28 1];

cnnConfig.layer{2}.type = 'conv';
cnnConfig.layer{2}.filterDim = [9 9];
cnnConfig.layer{2}.numFilters = 3;
cnnConfig.layer{2}.nonLinearType = 'sigmoid';
cnnConfig.layer{2}.conMatrix = ones(cnnConfig.layer{1}.dimension(3),cnnConfig.layer{2}.numFilters);

cnnConfig.layer{3}.type = 'pool';
cnnConfig.layer{3}.poolDim = [2 2];
cnnConfig.layer{3}.poolType = 'meanpool';

cnnConfig.layer{4}.type = 'conv';
cnnConfig.layer{4}.filterDim = [5 5];
cnnConfig.layer{4}.numFilters = 4;
cnnConfig.layer{4}.nonLinearType = 'sigmoid';
cnnConfig.layer{4}.conMatrix = ones(3,4);

cnnConfig.layer{5}.type = 'pool';
cnnConfig.layer{5}.poolDim = [2 2];
cnnConfig.layer{5}.poolType = 'meanpool';

cnnConfig.layer{6}.type = 'stack2line';

cnnConfig.layer{7}.type = 'sigmoid';
cnnConfig.layer{7}.dimension = 60;

cnnConfig.layer{8}.type = 'softmax';
cnnConfig.layer{8}.dimension = 10;

cnnConfig.costFun = 'crossEntropy';
end