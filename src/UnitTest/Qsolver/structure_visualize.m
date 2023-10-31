clear;
clc;
close all;

namelist = dir("*.csv");
for i = 1:length(namelist)
    load(namelist(i).name)
end

visual(r1, 1, x, y)
visual(r2, 2, x, y)
visual(w, 3, x, y)


function visual(f,n,x,y)
    figure(n)
    contourf(x,y,f,20,"linestyle","none")
    colormap(jet)
    colorbar();
    daspect([1 1 1])
end
