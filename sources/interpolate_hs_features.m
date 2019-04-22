function [features_hs_interpolated] = interpolate_hs_features(features_hs, spatial_dimensions, p, method)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

if (nargin == 3)
    method = 'nearest';
end

N1_HS = spatial_dimensions(1);
N2_HS = spatial_dimensions(2);
num_filters = size(features_hs,1);
features_hs = features_hs';

features_hs_interpolated = zeros(num_filters, N1_HS*N2_HS*p*p);
for i = 1:num_filters
    temp_lr = reshape(features_hs(:,i),[N1_HS N2_HS]);
    temp_hr = imresize(temp_lr, p,'Method',method);
    features_hs_interpolated(i,:) = temp_hr(:)'; 
end

features_hs_interpolated = features_hs_interpolated/max(features_hs_interpolated(:));
end

