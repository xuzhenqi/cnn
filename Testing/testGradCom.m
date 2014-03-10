cnnConfig = configTestGradient();
%  calling cnnInitParams() to initialize parameters
[db_theta, meta] = cnnInitParams(cnnConfig);

% Load MNIST Data
images = loadMNISTImages('../Dataset/MNIST/train-images-idx3-ubyte');
d1 = cnnConfig.layer{1}.dimension(1);
d2 = cnnConfig.layer{1}.dimension(2);
d3 = cnnConfig.layer{1}.dimension(3);

images = reshape(images,d1,d2,d3,[]);
labels = loadMNISTLabels('../Dataset/MNIST/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

db_images = images(:,:,:,1:10);
db_labels = labels(1:10);

[cost, grad] = cnnCost(db_theta,db_images,db_labels,cnnConfig,meta);


% Check gradients
numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
    db_labels,cnnConfig,meta), db_theta);

% Use this to visually compare the gradients side by side
% disp([numGrad grad]);
grad = thetaChange(grad,meta,'vec2stack',cnnConfig);
numGrad = thetaChange(numGrad,meta,'vec2stack',cnnConfig);

diff = zeros(meta.numLayers,2);
for l = 2 : meta.numLayers
    if ~isempty(grad{l}.W)
        diff(l,1) = norm(numGrad{l}.W(:)-grad{l}.W(:))/norm(numGrad{l}.W(:)+grad{l}.W(:));
        % Should be small. In our implementation, these values are usually
        % less than 1e-9.
        %disp(diff);
        %assert(diff < 1e-9,...
            %'Difference too large. Check your gradient computation again\nlayer %d.W',l);
        
        diff(l,2) = norm(numGrad{l}.b(:)-grad{l}.b(:))/norm(numGrad{l}.b(:)+grad{l}.b(:));
        % Should be small. In our implementation, these values are usually
        % less than 1e-9.
        %disp(diff);
        %assert(diff < 1e-9,...
           % 'Difference too large. Check your gradient computation again\nlayer %d.b',l);
    end
end
diff