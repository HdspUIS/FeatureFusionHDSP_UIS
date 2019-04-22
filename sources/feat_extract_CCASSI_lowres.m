function X = feat_extract_CCASSI_lowres(shot_data, cca, dimensions, shots, num_filters, filter_type, filter_set, dec_type, dec_factor)

N1  = dimensions(1);
N2  = dimensions(2);
L   = dimensions(3);

if strcmp(dec_type,'MS-image')
    % Spatial decimation matrix
    p       = dec_factor;
    N1_hs   = N1/p;
    
    p_vector= [];
    for i =1:p
        p_vector = [p_vector ((0:p-1) + (N1)*(i-1))];
    end
    
    a2 = [];
    b2 = [];
    for k = 1 : N1_hs
        a0 = 1 : N1_hs;
        a0 = a0 + (k-1) * N1_hs;
        a1 = repmat(a0,[p*p 1]);
        a3 = (0:(N1_hs -1))*p + 1;
        a3 = repmat(a3 + (k-1)*N1_hs*p*p,[p*p 1]);
        b0 = a3  + repmat(p_vector',[1 N1_hs]);
        a2 = [a2; a1(:)];
        b2 = [b2; b0(:)];
    end
    
    a = [];
    b = [];
    for j = 1:L
        at = a2 + (j-1)*N1_hs*N1_hs;
        bt = b2 + (j-1)*N1*N1;
        a = [a; at];
        b = [b; bt];
    end
    D=sparse(a,b,1,N1_hs*N1_hs*L,N1*N1*L); % Spatial decimation matrix
    
    % Sensing Matrix
    H = [];
    for j=1:shots
        Ad = sensing_matrix(N1,1,L,cca{j},N1 * (N1 + L - 1));
        H = [H; Ad];
    end
    
    B         = wav2_dct1(N1_hs, N1_hs, L);
    val       = powr_mthd(H, [N1*N1*L 1], 1e-6, 100, 0);
    H         = H./sqrt(val);
    val       = powr_mthd(D', [N1_hs*N1_hs*L 1], 1e-6, 100, 0);
    D         = D./sqrt(val);
    
    y = shot_data(:)/norm(shot_data(:)); % measurements
    
    lmb_max   = norm(B*(D * H'*y), inf); % maximo lambda
    Irec     = linz_admm_dec_matrix(H, D, B', y, 0.01*lmb_max, [], zeros(N1_hs,N1_hs,L));
    Irec     = reshape(Irec, N1_hs, N1_hs, L);
    
    Ir = imresize3(Irec,[N1 N2 L]); 
    
elseif strcmp(dec_type,'HS-image')
    q = dec_factor;
    L_ms = L/q;
    
    for i =1:q
        q_vector(i) = (i-1)*N1*N2;
    end
    
    a = [];
    b = [];
    for j = 1:L_ms
        a0 = 1:(N1 * N2);
        a1 = a0 + (j-1)*(N1*N2);
        a1 = repmat(a1,[q 1]);
        b1 = a0 + (j-1)*q*(N1*N2) + repmat(q_vector',[1 N1*N2]);
        a = [a; a1(:)];
        b = [b; b1(:)];
    end
    D=sparse(a,b,1,N1*N2*L_ms,N1*N2*L);

    % Sensing Matrix
    H = [];
    for j=1:shots
        Ad = sensing_matrix(N1,1,L,cca{j},N1 * (N1 + L - 1));
        H = [H; Ad];
    end
    
    B         = wav2_dct1(N1, N2, L_ms);
    val       = powr_mthd(H, [N1*N1*L 1], 1e-6, 100, 0);
    H         = H./sqrt(val);
    val       = powr_mthd(D', [N1*N1*L_ms 1], 1e-6, 100, 0);
    D         = D./sqrt(val);
    y         = shot_data(:)/norm(shot_data(:)); % measurements
    
    lmb_max  = norm(B*(D * H'*y), inf); % maximo lambda
    Irec     = linz_admm_dec_matrix(H, D, B', y, 0.01*lmb_max, [], zeros(N1,N2,L_ms));
    Irec     = reshape(Irec, N1, N2, L_ms);
    Ir       = imresize3(Irec,[N1 N2 L]); 
end


measure_Ir = zeros(size(shot_data));
for i = 1 : shots
    Ir_coded = cca{i} .* Ir;
    shifted_Ircoded{i} = zeros(N1,N1+L-1,L);
    for j = 1:L
        shifted_Ircoded{i}(:,j:N2+j-1,j) = Ir_coded(:,:,j);
    end
    measure_Ir(:,:,i) = reshape(sum(shifted_Ircoded{i},3),N1,N1+L-1);
end

% Fast version
for k = 1 : shots
    u1 = repmat(measure_Ir(:,:,k),[1 1 L]);
    u2 = repmat(shot_data(:,:,k),[1 1 L]);
    u3{k} = u2 .* (shifted_Ircoded{k}./u1);
end

X = zeros(num_filters, N1 * N2);
for j = 1 : N2
    for i = 1 : N1
        for k = 1:shots
            pos   = find(filter_set(:,filter_type(i,j,k)));
            m1=[];
            for l = 1:length(pos)
                m1(l) = u3{k}(i,j + pos(l) - 1, pos(l));
            end
            X(filter_type(i,j,k), (j-1)*N1 + i) = sum(m1(:));
        end
    end
end

end