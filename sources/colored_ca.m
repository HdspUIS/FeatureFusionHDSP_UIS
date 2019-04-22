function [cca_code, filter_disp, filter_set, filter_pos] = colored_ca(shots, N1, N2, L, num_filters, filtertype)

if (nargin == 5)
    filtertype = 'bandpass';
end

b1 = floor(shots / num_filters);
b2 = mod(shots, num_filters);

% Building the filter responses;
filter_wide = floor(L / num_filters);
c1 = mod(L,num_filters);
filter_set = zeros(L, num_filters);
filter_pos = zeros(2, num_filters);

base1 = (num_filters - c1) * filter_wide;
filter_wide2 = (L - base1) / c1;

if strcmp(filtertype,'binary')
    for i = 1:num_filters
        if (i <= num_filters - c1)
            d1 = (i-1) * filter_wide + 1;
            d2 = i * filter_wide;
            filter_set(d1:d2,i) = 1;
        else
            d1 = (i-1) * filter_wide + 1 + (i - (num_filters - c1) - 1);
            d2 = d1 + (filter_wide);
            filter_set(d1:d2,i) = 1;
        end
        filter_pos(:,i) =[d1; d2];
    end
    
elseif strcmp(filtertype,'bandpass')
    for i = 1:num_filters
        if (i <= num_filters - c1)
            d1 = ((i-1) * filter_wide) / L;
            d2 = ((i * filter_wide)) / L;
            if (i==1)
                fr = abs(fft(fir1(2*L,d2)));
            elseif (i == num_filters)
                fr = abs(fft(fir1(2*L,d1,'high')));
            else
                fr = abs(fft(fir1(2*L,[d1 d2])));
            end
            filter_set(:,i) = fr(1:L);
        else
            ii = i - (num_filters - c1);
            d1 = (base1 + ((ii-1) * filter_wide2)) / L;
            d2 = (base1 + (ii * filter_wide2)) / L;
            if (i==1)
                fr = abs(fft(fir1(2*L,d2)));
            elseif (i == num_filters)
                fr = abs(fft(fir1(2*L,d1,'high')));
            else
                fr = abs(fft(fir1(2*L,[d1 d2])));
            end
            filter_set(:,i) = fr(1:L);
        end
    end
    
elseif strcmp(filtertype,'random-uniform')
    filter_set = rand(L, num_filters);   
end

% Normalization stage
for i = 1:size(filter_set,1)
    filter_set(i,:) = filter_set(i,:)/sum(abs(filter_set(i,:)));
end



filter_type = 1:num_filters;
filter_disp = zeros(N1,N2,shots);
if (b1 > 0) && (b2 ~= 0)
    for i = 1 : N1
        for j = 1 : N2
            filter_base = repmat(filter_type,1,b1);
            rp1 = randperm(num_filters);
            rp2 = randperm(num_filters*b1);
            filter_disp(i,j,:) = [filter_base(rp2)'; filter_type(rp1(1:b2))'];
        end
    end
end

if (b1 > 0) && (b2 == 0)
    for i = 1 : N1
        for j = 1 : N2
            filter_base = repmat(filter_type,1,b1);
            rp2 = randperm(num_filters*b1);
            filter_disp(i,j,:) = filter_base(rp2)';
        end
    end
end

if (b1 == 0)
    for i = 1 : N1
        for j = 1 : N2
            rp1 = randperm(num_filters);
            filter_disp(i,j,:) = filter_type(rp1(1:b2))';
        end
    end
end


for i = 1:shots
    cca_code{i} = zeros(N1,N2,L);
end

for i = 1:N1
    for j = 1:N2
        for k = 1: shots
            cca_code{k}(i,j,:) = filter_set(:,filter_disp(i,j,k))'; 
        end
    end
end