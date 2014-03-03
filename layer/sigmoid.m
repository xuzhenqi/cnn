function h=sigmoid(a)
%sigmoid computes the sigmoid of matrix a
% 
% Parameters:
%  a : could be scale, vector, matrix, multiple dimetional array;
% 
% Returns:
%  h : the same size as a.

  h=1./(1+exp(-a));
end
