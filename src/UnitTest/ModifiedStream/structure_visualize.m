clear;
clc;
close all;

namelist = dir("*.csv");
for i = 1:length(namelist)
    load(namelist(i).name)
end

% visual(I0h11,1,x,y)
% visual(h11exact,2,x,y)
% visual(I0h11-h11exact,3,x,y)
% visual(I0h12-h12exact,4,x,y)
% visual(wnlexact,5,x,y)
 visual(I0wnonl-wnlexact,6,x,y)
visual(I0r1nonl-r1nlexact,7,x,y)
visual(I0r2nonl-r2nlexact,8,x,y)
% visual(I0u-uexact,1,x,y)
% visual(I0v-vexact,2,x,y)
% visual(I0w-wexact,3,x,y)

% visual(p11 -p11e,7,x,y)
% visual(p12-p12e,8,x,y)
% visual(p21-p21e,9,x,y)
% visual(Dyy_p21_+1.4*4*sin(2*x+y),10,x,y)
% figure(15)
% contourf(x,y,Dyy_p21_+1.4*4*sin(2*x+y))
% visual(convectw,10,x,y)
% visual(I0wnonl-wnlexact,11,x,y)
% title("w")
% visual(I0r1nonl-r1nlexact,12,x,y)
% title("r1")
% visual(I0r2nonl-r2nlexact,13,x,y)
% title("r2")
%visual(I0u-uexact,1,x,y)

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