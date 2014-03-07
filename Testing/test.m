x = cell(5,1);
x{2}.W = ones(5);
x{2}.b = 5;
x{3}.b = zeros(6);
x{3}.W = 6;
y = [x{2}.W(:);x{2}.b(:)];