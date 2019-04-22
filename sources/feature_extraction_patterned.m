function X = feature_extraction_patterned(shot_data, dimensions, shots, num_filters, filter_type)

N1  = dimensions(1);
N2  = dimensions(2);
% L   = dimensions(3);

X = zeros(num_filters, N1 * N2);
for i = 1 : N2
    basic_indexes = (((i-1)*N1 + 1: i*N1)' - 1)*num_filters;
    for k = 1:shots
        X(basic_indexes + filter_type(:,i,k)) = shot_data(:,i,k);
    end
end