%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Routine: Fused feature visualization for different values of the TV
%   regularization parameter (tau2)
%
%   Ramirez, J., & Arguello, H. (2019). Multiresolution Compressive Feature
%   Fusion for Spectral Image Classification
%
%   Author:
%   Dr. Juan Marcos Ramirez
%   Universidad Industrial de Santander, Bucaramanga, Colombia
%   Universidad de Los Andes, Merida, Venezuela
%   email: juanra@ula.ve, juanmarcos26@gmail.com
%
%   Date: April, 2019
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;

addpath('sources/');
addpath(genpath('external_code/'));

disp('---------------------------------------');
disp('This routine could take several minutes');
disp('---------------------------------------');

%% Loading data
load('data/PaviaU.mat');
Io = paviaU(end-255:end,1:256,1:1:96);
clear paviaU;

load('data/PaviaU_gt.mat');
ground_truth = paviaU_gt(end-255:end,1:256);
clear paviaU_gt;

L   = size(Io,3);
N1  = size(Io,1);
N2  = size(Io,2);

% Ground truth correction
for i = 1:N1
    for j = 1:N2
        if (ground_truth(i,j) > 5)
            ground_truth(i,j) = ground_truth(i,j) - 1;
        end
    end
end

%% Image downsampling
% spatial downsampling (building the hyperspectral image)
p       = 4;
window  = 5;
sigma   = 1.00;
I_hs    = spatial_blurring(Io, p, window, sigma,'sum');

% spectral downsampling (building the multispectral image)
q = 4;
I_ms = spectral_blurring(Io, q,'sum');
%% CSI shots (capturing compressive measurements)
compression_ratio   = 0.25;
[N1_HS, N2_HS,L_HS] = size(I_hs);
[N1_MS, N2_MS,L_MS] = size(I_ms);

% 3D-CASSI measurements
[shot_patt_hs, shots_hs, num_hs_filters, ~, filter_patt_hs, filter_set_patt_hs] = ...
    patterned_shots(I_hs, compression_ratio,'binary');

[shot_patt_ms, shots_ms, num_ms_filters, ~, filter_patt_ms] = ...
    patterned_shots(I_ms, compression_ratio,'binary');

% C-CASSI measurements
[shot_ccassi_hs, ~, ~, cca_hs, filter_type_hs, ~, filter_set_hs, filter_pos_HS] =...
    cassi_shots(I_hs, compression_ratio,'binary');

[shot_ccassi_ms, ~, ~, cca_ms, filter_type_ms, ~, filter_set_ms, filter_pos_MS] =...
    cassi_shots(I_ms, compression_ratio,'binary');
%% Feature extraction from HS compressive measurements

% Features using the 3D-CASSI architecture 
features_hs = feature_extraction_patterned(shot_patt_hs, size(I_hs),...
    shots_hs, num_hs_filters, filter_patt_hs);
clear shot_patt_hs filter_patt_hs;

features_ms = feature_extraction_patterned(shot_patt_ms, size(I_ms),...
    shots_ms, num_ms_filters, filter_patt_ms);
clear shot_patt_ms filter_patt_ms;

% Features using the C-CASSI architecture
features_hs_ccassi = feat_extract_CCASSI_lowres(shot_ccassi_hs, cca_hs,...
    size(I_hs), shots_hs, num_hs_filters, filter_type_hs, filter_set_hs, 'HS-image', 1);
clear shot_ccassi_hs cca_hs filter_type_hs;

features_ms_ccassi = feat_extract_CCASSI_lowres(shot_ccassi_ms, cca_ms,...
    size(I_ms), shots_ms, num_ms_filters, filter_type_ms, filter_set_ms, 'MS-image', 1);
clear shot_ccassi_ms cca_ms filter_type_ms;
%% Feature Fusion

% Target high-resolution features
high_resolution_features = reshape(Io,[N1*N2 L]);
high_resolution_features = filter_set_patt_hs' * high_resolution_features';
high_resolution_features = reshape(high_resolution_features',[N1 N2 num_hs_filters]);

% Proposed feature fusion using the patterned architecture
features_patt_ms_cube = reshape(features_ms',[N1_MS N2_MS num_ms_filters]);
features_patt_hs_cube = reshape(features_hs',[N1_HS N2_HS num_hs_filters]);

disp('(1/6) Feature fusion using 3D-CASSI tau = 1e-5');
tic;
fused_features_patt1 = feature_fusion(features_patt_ms_cube,features_patt_hs_cube,q,p,1e-5);
fused_features_patt1 = reshape(fused_features_patt1',[N1 N2 num_hs_filters]);
toc;
disp('---------------------------------------');
disp('(2/6) Feature fusion using 3D-CASSI tau = 1e-4');
tic;
fused_features_patt2 = feature_fusion(features_patt_ms_cube,features_patt_hs_cube,q,p,1e-4);
fused_features_patt2 = reshape(fused_features_patt2',[N1 N2 num_hs_filters]);
toc;
disp('---------------------------------------');
disp('(3/6) Feature fusion using 3D-CASSI tau = 1e-3');
tic;
fused_features_patt3 = feature_fusion(features_patt_ms_cube,features_patt_hs_cube,q,p,1e-3);
fused_features_patt3 = reshape(fused_features_patt3',[N1 N2 num_hs_filters]);
toc;
disp('---------------------------------------');


% Proposed feature fusion using the C-CASSI architecture
features_ccassi_ms_cube = reshape(features_ms_ccassi',[N1_MS N2_MS num_ms_filters]);
features_ccassi_hs_cube = reshape(features_hs_ccassi',[N1_HS N2_HS num_hs_filters]);

disp('(4/6) Feature fusion using 3D-CASSI tau = 1e-5');
tic;
fused_features_cassi1 = feature_fusion(features_ccassi_ms_cube,features_ccassi_hs_cube,q,p,1e-5);
fused_features_cassi1 = reshape(fused_features_cassi1',[N1 N2 num_hs_filters]);
toc;
disp('---------------------------------------');
disp('(5/6) Feature fusion using 3D-CASSI tau = 1e-4');
tic;
fused_features_cassi2 = feature_fusion(features_ccassi_ms_cube,features_ccassi_hs_cube,q,p,1e-4);
fused_features_cassi2 = reshape(fused_features_cassi2',[N1 N2 num_hs_filters]);
toc;
disp('---------------------------------------');
disp('(6/6) Feature fusion using 3D-CASSI tau = 1e-3');
tic;
fused_features_cassi3 = feature_fusion(features_ccassi_ms_cube,features_ccassi_hs_cube,q,p,1e-3);
fused_features_cassi3 = reshape(fused_features_cassi3',[N1 N2 num_hs_filters]);
toc;
disp('---------------------------------------');

%% Display results
close all

HRI = high_resolution_features(:,:,end);
HRI = HRI/max(HRI(:));

normColor = @(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
band_set_Io=[floor(2*L/3) floor(L/4) 1];
temp_show=Io(:,:,band_set_Io);temp_show=normColor(temp_show);

figure('units','normalized','outerposition',[0 0 1 1])
subplot(331);
imshow(temp_show);
xlabel('(a)');
title('HR image');

subplot(332)
imshow(label2rgb(ground_truth));
xlabel('(b)')
title('Ground truth');

subplot(333);
imshow(imadjust(HRI),[])
xlabel('(c)')
title('Band of high-resolution features');

subplot(334);
imshow(imadjust(fused_features_patt1(:,:,end)),[]);
xlabel('(d) \tau_2 = 1e-5')
title('Band of fused features (3D-CASSI)');

subplot(335);
imshow(imadjust(fused_features_patt2(:,:,end)),[]);
xlabel('(e) \tau_2 = 1e-4')
title('Band of fused features (3D-CASSI)');

subplot(336);
imshow(imadjust(fused_features_patt3(:,:,end)),[]);
xlabel('(f) \tau_2 = 1e-3')
title('Band of fused features (3D-CASSI)');

subplot(337);
imshow(imadjust(fused_features_cassi1(:,:,end)),[]);
xlabel('(g) \tau_2 = 1e-5')
title('Band of fused features (C-CASSI)');

subplot(338);
imshow(imadjust(fused_features_cassi2(:,:,end)),[]);
xlabel('(h) \tau_2 = 1e-4')
title('Band of fused features (C-CASSI)');

subplot(339);
imshow(imadjust(fused_features_cassi3(:,:,end)),[]);
xlabel('(i) \tau_2 = 1e-3')
title('Band of fused features (C-CASSI)');