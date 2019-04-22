classdef wav2_dct1
    properties ( Access = private )
        adj        
        Nx_c
        Ny_c
        Nz_c
        coef_v
    end
    
    methods
        function  res   = wav2_dct1(Nx, Ny, Nz)
  
            res.adj     = 0;
            res.Nx_c    = Nx;
            res.Ny_c    = Ny;
            res.Nz_c    = Nz;  

            res.coef_v  = MakeONFilter('Symmlet', 8);
        end
        
        function  res   = ctranspose(obj)
            obj.adj     = xor(obj.adj, 1);
            res         = obj;
        end
        
        function  out   = mtimes(obj, int)
            Nx          = obj.Nx_c;
            Ny          = obj.Ny_c;
            Nz          = obj.Nz_c;
            coef        = obj.coef_v;
            
           if obj.adj == 1  % Psi*x
                X = reshape(int, Nx*Ny, Nz)';
                U = dct(X)';
                U = reshape(U, Nx, Ny, Nz);
                Y = U;
                for nz = 1:Nz
                    Y(:, :, nz) = FWT2_PO(U(:, :, nz), 0, coef);
                end
                out = Y(:);
            else     % Psi'*x
                X = reshape(int, Nx, Ny, Nz); 
                U = X;
                for nz = 1:Nz
                    U(:, :, nz) = IWT2_PO(X(:, :, nz), 0, coef);
                end
                U = reshape(U, Nx*Ny, Nz)';
                Y = idct(U)';
                out = Y(:);
            end         
        end
    end
end