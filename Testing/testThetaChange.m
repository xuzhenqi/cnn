% testThetaChange
cnnConfig = config();
[Theta, meta] = cnnInitParams(cnnConfig);
newTheta = thetaChange(Theta, meta, 'vec2stack', cnnConfig);
oldTheta = thetaChange(newTheta, meta, 'stack2vec', cnnConfig);
newOldTheta = thetaChange(oldTheta, meta, 'vec2stack', cnnConfig);

if isequal(newTheta,newOldTheta);
    fprintf('The thetaConfig function works well, congratulations\n');
else
    fprintf('The thetaConfig function does not work well, please check!\n');
end

if isequal(Theta,oldTheta);
    fprintf('The thetaConfig function works well, congratulations\n');
else
    fprintf('The thetaConfig function does not work well, please check!\n');
end