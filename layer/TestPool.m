Times = 50;
x = magic(1000);
tic;
for i = 1 : Times;
    y = cnnPool([3 3],x,'meanpool');
end
fprintf('meanpool time used:%s\n',toc);
for i = 1 : Times;
    y = cnnPool([3 3],x,'newpool');
end
fprintf('newpool time used:%s\n',toc);