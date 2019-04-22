function [shot_data, shots, num_filters, cca, filter_type, shot_code, filter_set, filter_pos] = cassi_shots(I, compression_ratio, filtertype)

if (nargin == 2)
    filtertype = 'bandpass';
end

[N1, N2, L] = size(I);
num_filters = floor(L * compression_ratio);
shots       = floor(L * compression_ratio);

% Multispectral measurements
[cca, filter_type, filter_set, filter_pos]  = colored_ca(shots, N1, N2, L, num_filters, filtertype);
shot_data = zeros(N1, N2 + L - 1, shots);
shot_code = zeros(N1, N2 + L - 1, shots);

for i = 1 : shots
    shot_data(:,:,i) = zeros(N1, N2 + L - 1);
    shot_code(:,:,i) = zeros(N1, N2 + L - 1);
%     dispersed{i} = zeros(N1, N2 + L - 1, L);
    for j = 1 : L
        temp1 = zeros(N1, N2 + L - 1);
        temp2 = zeros(N1, N2 + L - 1);
        temp1(:,j:N2 + j - 1) = I(:,:,j).*cca{i}(:,:,j);
        temp2(:,j:N2 + j - 1) = cca{i}(:,:,j);
        shot_data(:,:,i) =  shot_data(:,:,i) + temp1;
        shot_code(:,:,i) =  shot_code(:,:,i) + temp2;
%         dispersed{i}(:,j:N2 + j - 1,j) = cca{i}(:,:,j);
    end
end