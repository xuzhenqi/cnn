function [output, linTrans] = nonlinear(input,W,b,type,norm)
%sigmoid computes nonlinear transformation of the input
% 
% Parameters:
%  input : matrix, inputDim * Num
%  W     : matrix, OutputDim * inputDim
%  b     : vector, OutputDim * 1
%  type  : string, could be 'relu', 'sigmoid', 'tanh', 'softmax',
%  'softsign'
% 
% Returns:
%  output : matrix, OutputDim * Num

if ~exist('type','var')
    type = 'sigmoid';
end

linTrans = W * input + repmat(b,[1 size(input,2)]);
switch type
    case 'sigmoid'
        output = 1 ./ (1 + exp(- linTrans));
    case 'relu'
        output = max(0, linTrans);
    case 'tanh'
        output = tanh(linTrans);
    case 'softmax'
        % Attention: the softmax() in neural network toolbox can not be used here.
        expLinTrans = exp(linTrans);
        output = expLinTrans ./ repmat(sum(expLinTrans),size(expLinTrans,1),1);
    case 'softsign'
        output = linTrans ./ (1 + abs(linTrans));
end

if exist('norm','var')
    % Todo
end

end
