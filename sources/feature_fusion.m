function [fused_features] = feature_fusion(features_ms_image,features_hs_image,q,p,lambda)

% features_ms_image = reshape(features_ms',[N1_MS N2_MS num_ms_filters]);
% features_hs_image = reshape(features_hs',[N1_HS N2_HS num_hs_filters]);

[n1f,n2f,lf_ms]=size(features_ms_image);
[n1f_hs,n2f_hs,lf]=size(features_hs_image);

for i =1:q
    q_vector(i) = (i-1)*n1f*n2f;
end

a = [];
b = [];
for j = 1:lf_ms
    a0 = 1:(n1f * n2f);
    a1 = a0 + (j-1)*(n1f*n2f);
    a1 = repmat(a1,[q 1]);
    b1 = a0 + (j-1)*q*(n1f*n2f) + repmat(q_vector',[1 n1f*n2f]);
    a = [a; a1(:)];
    b = [b; b1(:)];
end
A1=sparse(a,b,1,n1f*n2f*lf_ms,n1f*n2f*lf);


p_vector = [];
for i =1:p
    p_vector = [p_vector ((0:p-1) + (n1f)*(i-1))];
end

a2 = [];
b2 = [];
for k = 1 : n2f_hs
    a0 = 1 : n1f_hs;
    a0 = a0 + (k-1) * n1f_hs;
    a1 = repmat(a0,[p*p 1]);
    a3 = (0:(n2f_hs -1))*p + 1;
    a3 = repmat(a3 + (k-1)*n1f_hs*p*p,[p*p 1]);
    b0 = a3  + repmat(p_vector',[1 n1f_hs]);
    a2 = [a2; a1(:)];
    b2 = [b2; b0(:)];
end

a = [];
b = [];
for j = 1:lf
    at = a2 + (j-1)*n1f_hs*n2f_hs;
    bt = b2 + (j-1)*n1f*n2f;
    a = [a; at];
    b = [b; bt];
end
A2=sparse(a,b,1,n1f_hs*n2f_hs*lf,n1f*n2f*lf);


% r_hs = fspecial('gaussian',5,2.50);
% R_hs = opConvolve(n1f, n2f, r_hs,[3,3],'truncated');
% [u1,u2] =size(R_hs);

H       = [A1; A2];
% B       = tv_mtrx_2(n1f, n2f, lf);
B       = wav2_dct1(n1f, n2f, lf);
% B       = wav2(n1f, n2f, lf);
Lo      = tv_mtrx_2(n1f, n2f, lf);
val     = powr_mthd(H, [n1f*n2f*lf 1], 1e-6, 100, 0);
H       = H./sqrt(val);
y       = [features_ms_image(:); features_hs_image(:)] / norm([features_ms_image(:); features_hs_image(:)]);

% lmb_max     = 0.01*norm(B*(H'*y), inf); % maximo lambda
% Irec        = linz_admm(H, B', y, lmb_max, [], zeros(n1f,n2f,lf));
% Irec        = linz_admm_TV(speye(length(Irec)), L, Irec, lambda, [], zeros(n1f,n2f,lf));

Irec        = linz_admm_L1TV(H, B', Lo, y, lambda, 0.01*norm(B*(H'*y),inf), [], zeros(n1f,n2f,lf));

Irec_fuse   = reshape(Irec, n1f, n2f, lf);
fused_features = reshape(Irec_fuse, [n1f*n2f lf])';
end

