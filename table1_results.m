%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Routine: Performance of the proposed feature fusion approach on the
%   Pavia University dataset
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

lambda              = 1e-4;
compression_ratio   = 0.25;
training_rate       = 0.10;
trials              = 5;

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

for i = 1:trials
    tic;
    disp(['Iteration: ' num2str(i)]);
    %% CSI shots (capturing compressive measurements)
    [N1_HS, N2_HS,L_HS] = size(I_hs);
    [N1_MS, N2_MS,L_MS] = size(I_ms);
    
    % Patterned measurements
    [shot_patt_hs, shots_hs, num_hs_filters, ~, filter_patt_hs] = ...
        patterned_shots(I_hs, compression_ratio,'binary');
    
    [shot_patt_ms, shots_ms, num_ms_filters, ~, filter_patt_ms] = ...
        patterned_shots(I_ms, compression_ratio,'binary');
    
    % C-CASSI measurements
    [shot_ccassi_hs, ~, ~, cca_hs, filter_type_hs, ~, filter_set_hs] =...
        cassi_shots(I_hs, compression_ratio,'binary');
    
    [shot_ccassi_ms, ~, ~, cca_ms, filter_type_ms, ~, filter_set_ms] =...
        cassi_shots(I_ms, compression_ratio,'binary');
    %% Feature extraction from HS compressive measurements
    % Features using the patterned architecture
    features_hs = feature_extraction_patterned(shot_patt_hs, size(I_hs),...
        shots_hs, num_hs_filters, filter_patt_hs);
    features_hs_hd = interpolate_hs_features(features_hs,[N1_HS N2_HS], p, 'nearest');
    
    features_ms = feature_extraction_patterned(shot_patt_ms, size(I_ms),...
        shots_ms, num_ms_filters, filter_patt_ms);
    
    % Features using the c-cassi architecture
    features_hs_ccassi = feat_extract_CCASSI_lowres(shot_ccassi_hs, cca_hs,...
        size(I_hs), shots_hs, num_hs_filters, filter_type_hs, filter_set_hs, 'HS-image', 4);
    features_hs_ccassi_hd = interpolate_hs_features(features_hs_ccassi,...
        [N1_HS N2_HS], p, 'nearest');
    
    features_ms_ccassi = feat_extract_CCASSI_lowres(shot_ccassi_ms, cca_ms,...
        size(I_ms), shots_ms, num_ms_filters, filter_type_ms, filter_set_ms, 'MS-image', 2);
    %% Feature Fusion
    disp('Fusing features (3D-CASSI)');
    % Proposed feature fusion using the patterned architecture
    features_patt_ms_cube = reshape(features_ms',[N1_MS N2_MS num_ms_filters]);
    features_patt_hs_cube = reshape(features_hs',[N1_HS N2_HS num_hs_filters]);
    fused_features_patt_clss = feature_fusion(features_patt_ms_cube,features_patt_hs_cube,q,p,lambda);
    
    disp('Fusing features (C-CASSI)');
    % Proposed feature fusion using the C-CASSI architecture
    features_cssi_ms_cube = reshape(features_ms_ccassi',[N1_MS N2_MS num_ms_filters]);
    features_cssi_hs_cube = reshape(features_hs_ccassi',[N1_HS N2_HS num_hs_filters]);
    fused_features_cssi_clss = feature_fusion(features_cssi_ms_cube,features_cssi_hs_cube,q,p,lambda);
    %% Classification Stage
    [training_indexes, test_indexes] = classification_indexes(ground_truth, training_rate);
    T_classes = ground_truth(training_indexes);
    
    stack_features_patt       = [features_ms; features_hs_hd];
    stack_features_patt_train = stack_features_patt(:,training_indexes);
    stack_features_patt_test  = stack_features_patt(:,test_indexes);
    
    stack_features_cssi       = [features_ms_ccassi; features_hs_ccassi_hd];
    stack_features_cssi_train = stack_features_cssi(:,training_indexes);
    stack_features_cssi_test  = stack_features_cssi(:,test_indexes);
    
    fused_features_patt_train = fused_features_patt_clss(:,training_indexes);
    fused_features_patt_test  = fused_features_patt_clss(:,test_indexes);
    
    fused_features_cssi_train = fused_features_cssi_clss(:,training_indexes);
    fused_features_cssi_test  = fused_features_cssi_clss(:,test_indexes);
    
    raw_image_features          = reshape(Io,[N1*N2 L])';
    raw_image_features_train    = raw_image_features(:,training_indexes);
    raw_image_features_test     = raw_image_features(:,test_indexes);
    
    disp(['Predicting classes using the SVM-PLY classification approach...']);
    t          = templateSVM('KernelFunction','poly','Standardize',1,'KernelScale','auto');
    
    Mdl1         = fitcecoc(stack_features_patt_train',T_classes,'Learners',t);
    class_hat1   = predict(Mdl1, stack_features_patt_test');
    
    Mdl2         = fitcecoc(stack_features_cssi_train',T_classes,'Learners',t);
    class_hat2   = predict(Mdl2, stack_features_cssi_test');
    
    Mdl3         = fitcecoc(fused_features_patt_train',T_classes,'Learners',t);
    class_hat3   = predict(Mdl3, fused_features_patt_test');
    
    Mdl4         = fitcecoc(fused_features_cssi_train',T_classes,'Learners',t);
    class_hat4   = predict(Mdl4, fused_features_cssi_test');
    
    Mdl5         = fitcecoc(raw_image_features_train',T_classes,'Learners',t);
    class_hat5   = predict(Mdl5, raw_image_features_test');
    
    % Building the classification maps
    training_set_image = zeros(size(ground_truth,1), size(ground_truth,2));
    training_set_image(training_indexes) = ground_truth(training_indexes);
    
    image_stackP    = class_map_image(ground_truth, class_hat1, training_indexes, test_indexes);
    image_stackC    = class_map_image(ground_truth, class_hat2, training_indexes, test_indexes);
    image_fusionP   = class_map_image(ground_truth, class_hat3, training_indexes, test_indexes);
    image_fusionC   = class_map_image(ground_truth, class_hat4, training_indexes, test_indexes);
    image_raw_image = class_map_image(ground_truth, class_hat5, training_indexes, test_indexes);
    
    [OA1(i), AA1(i), kappa1(i), acc1(:,i)] = compute_accuracy(ground_truth(test_indexes), uint8(image_stackP(test_indexes)));
    [OA2(i), AA2(i), kappa2(i), acc2(:,i)] = compute_accuracy(ground_truth(test_indexes), uint8(image_stackC(test_indexes)));
    [OA3(i), AA3(i), kappa3(i), acc3(:,i)] = compute_accuracy(ground_truth(test_indexes), uint8(image_fusionP(test_indexes)));
    [OA4(i), AA4(i), kappa4(i), acc4(:,i)] = compute_accuracy(ground_truth(test_indexes), uint8(image_fusionC(test_indexes)));
    [OA5(i), AA5(i), kappa5(i), acc5(:,i)] = compute_accuracy(ground_truth(test_indexes), uint8(image_raw_image(test_indexes)));
    toc;
    disp('---------------------------------------');
end

macc1 = 100*mean(acc1,2);
macc2 = 100*mean(acc2,2);
macc3 = 100*mean(acc3,2);
macc4 = 100*mean(acc4,2);
macc5 = 100*mean(acc5,2);

mOA1  = 100*mean(OA1);
mOA2  = 100*mean(OA2);
mOA3  = 100*mean(OA3);
mOA4  = 100*mean(OA4);
mOA5  = 100*mean(OA5);

mAA1  = 100*mean(AA1);
mAA2  = 100*mean(AA2);
mAA3  = 100*mean(AA3);
mAA4  = 100*mean(AA4);
mAA5  = 100*mean(AA5);

mkappa1 = mean(kappa1);
mkappa2 = mean(kappa2);
mkappa3 = mean(kappa3);
mkappa4 = mean(kappa4);
mkappa5 = mean(kappa5);

%% Display results
disp('------------------------------------------------------------------------------');
fprintf('Class    \t HR      \t Stacking   \t Stacking   \t Fusion   \t Fusion \n');
fprintf('         \t image   \t C-CASSI    \t 3D-CASSI   \t C-CASSI  \t 3D-CASSI \n');
disp('------------------------------------------------------------------------------');
fprintf('Asphalt  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f \n',macc5(1), macc2(1), macc1(1), macc4(1), macc3(1));
fprintf('Meadows  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f \n',macc5(2), macc2(2), macc1(2), macc4(2), macc3(2));
fprintf('Gravel   \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f \n',macc5(3), macc2(3), macc1(3), macc4(3), macc3(3));
fprintf('Trees    \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f \n',macc5(4), macc2(4), macc1(4), macc4(4), macc3(4));
fprintf('Bare Soil\t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f \n',macc5(5), macc2(5), macc1(5), macc4(5), macc3(5));
fprintf('Bitumen  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f \n',macc5(6), macc2(6), macc1(6), macc4(6), macc3(6));
fprintf('Bricks   \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f \n',macc5(7), macc2(7), macc1(7), macc4(7), macc3(7));
fprintf('Shadows  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f  \t %3.2f \n',macc5(8), macc2(8), macc1(8), macc4(8), macc3(8));
disp('------------------------------------------------------------------------------');
fprintf('OA       \t %3.2f   \t %3.2f   \t %3.2f   \t %3.2f   \t %3.2f \n',mOA5, mOA2, mOA1, mOA4, mOA3);
fprintf('AA       \t %3.2f   \t %3.2f   \t %3.2f   \t %3.2f   \t %3.2f \n',mAA5, mAA2, mAA1, mAA4, mAA3);
fprintf('Kappa    \t %1.3f   \t %1.3f   \t %1.3f   \t %1.3f   \t %1.3f \n',mkappa5, mkappa2, mkappa1, mkappa4, mkappa3);
disp('------------------------------------------------------------------------------');