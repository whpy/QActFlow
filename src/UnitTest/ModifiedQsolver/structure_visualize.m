clear;
clc;
close all;
% 
% namelist = dir("*.csv");
% for i = 1:length(namelist)
%     load(namelist(i).name)
% end
load("x.csv")
load("y.csv")
n = 6000
w = load(num2str(n)+"w.csv");

r1 = load(num2str(n)+"r2.csv");

r2 = load(num2str(n)+"r1.csv");

S = load(num2str(n)+"S.csv");
visual(r1,1,x,y)
caxis([-0.5 0.5])

visual(r2,2,x,y)
caxis([-0.5 0.5])

visual(w,3,x,y)
caxis([-2 2])

visual(S,4,x,y)
figure(4)
colormap(hot)
caxis([0 1])

function visual(f,n,x,y)
    figure(n)
    contourf(x,y,f,20,"linestyle","none")
    colormap(jet)
    colorbar();
    daspect([1 1 1])
end
