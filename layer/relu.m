function [ y ] = relu( x )
%relu implements rectified linear units 
%   Params:
%       x: input
%       y: output, the same size as x

y = max(0,x);
end

