clear all;
close all;

G = MakeONFilter('Symmlet',8);

row = 128;
col = 128;
level = 1;
wc = zeros(row,col);

for i = 1:row;
    i
    for j = 1:col;        
        wc(i,j) = 1;
        iw = IWT2_PO(wc,level,G);
        A(:,(i-1)*row + j) = single(iw(:));
        wc(i,j) = 0;
    end
end
        