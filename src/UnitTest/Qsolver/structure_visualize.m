clear;
clc;
close all;

namelist = dir("*.csv");
for i = 1:length(namelist)
    load(namelist(i).name)
end

visual(cross1, 1, x, y)
visual(crossa, 2, x, y)



function visual(f,n,x,y)
    figure(n)
    contourf(x,y,f,20,"linestyle","none")
    colormap(jet)
    colorbar();
    daspect([1 1 1])
end
