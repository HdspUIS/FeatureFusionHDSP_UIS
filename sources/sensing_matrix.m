function A=sensing_matrix(N,shift,H,dmd,Sh)
    
img = zeros(N,N+shift*(H-1));

for k = 1:H
    img(1:N,shift*(k-1)+1:N+shift*(k-1),k) = dmd(:,:,k);
end


N2=Sh/N;

at=[];
bt=[];
for r=1:H
    [a,b]=find(img(:,:,r));
    ax=a(:)+(b(:)-1)*N;
    bx=(r-1)*N^2+(b(:)-1-shift*(r-1))*N+a(:);
    at=[at;ax(:)];
    bt=[bt;bx(:)];
end
A=sparse(at,bt,1,Sh,N^2*H);