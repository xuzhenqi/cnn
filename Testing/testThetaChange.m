% testThetaChange
cnnConfig = config();
[Theta meta] = cnnInitParams(cnnConfig);
newTheta = thetaChange(Theta, meta, 'stack2vec', cnnConfig);
oldTheta = thetaChange(newTheta, meta, 'vec2stack', cnnConfig);

if newTheta == oldTheta
    fprintf('The thetaConfig function works well, congratulations');
else
    fprintf('The thetaConfig function does not work well, please check!');
end
