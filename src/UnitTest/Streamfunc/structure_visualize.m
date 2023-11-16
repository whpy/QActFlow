clear;
clc;
close all;

namelist = dir("*.csv");
for i = 1:length(namelist)
    load(namelist(i).name)
end


visual(nl0a-nl0, 9, x, y)
figure(9)
title("w")
visual(nl1a-nl1, 10, x, y)
figure(10)
title("r1")
visual(nl2a-nl2, 11, x, y)
figure(11)
title("r2")

% visual(1.15349*p12a-p12, 12, x, y)
% visual(1.15349*p21a-p21, 13, x, y)
% visual(1.15349*p11a-p11, 14, x, y)
function visual(f,n,x,y)
    figure(n)
    contourf(x,y,f,20,"linestyle","none")
    colormap(jet)
    colorbar();
    daspect([1 1 1])
end
% figure(4);
% ana = r;
% for i = 1:size(x,1)
%     for j = 1:size(y,2)
%         r2 = (x(i,j)-pi)^2 + (y(i,j)-pi)^2;
%         ana(i,j) = exp(-r2/0.2);
%     end
% end
% contourf(x,y,ana,20,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("analytic solution")

% figure(4);
% err = ana-r;
% contourf(x,y,err,20,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("err distribution")