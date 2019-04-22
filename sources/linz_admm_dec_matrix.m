function x = linz_admm_dec_matrix(H, D, B, y, lmb, parm, sol)
% ========================================================
%   Solves the following problem via linearized ADMM:
%
%   History is a structure that contains the objective value, the primal and
%   dual residual norms, and the tolerances for the primal and dual residual
%   norms at each iteration.
%
%   min_{f, x}
%   (1/2)||Hf - y||_2^2 + \lmb||x||_1
%   sub. to   
%   Df - x = 0 
%
%   ===== Inputs  ===== 
%   A:          Forward measurement operator.
%   D:          Difference vertical and horizontal operator.
%   y:          Measurement data vector.
%   lmb:        
%   data:       Hyperspectral datacube to be used   
%   parm:       Structure containing parameters for the solver 
%               and defined as follows.
%
%               y: Input data (measurements).
%
%               epsilon: Noise bound.
%
%               
%
% 	===== Outputs =============
%   result:     Reconstructed hyperspectral datacube
%   time:       Algorithm execution time
%   psnr:       Peak-Signal-to-Noise_Ratio between original and
%               reconstructed datacubes
%
%   alpha is the over-relaxation parameter (typical values for alpha are
%   between 1.0 and 1.8).
%
% ========================================================

if ~isfield(parm,  'tol'), TOL  = 1e-6;
else, TOL    = parm.tol;  end
if ~isfield(parm, 'prnt'), PRNT = 0;    
else, PRNT   = parm.prnt; end
if ~isfield(parm, 'mitr'), MITR = 200;  
else, MITR   = parm.mitr; end
if ~isfield(parm,  'rho'), rho  = 0.1;  
else, rho    = parm.rho;  end

% save a matrix-vector multiply
[N, ~, L]  = size(sol);
Hty        = D * H' * y;
Hf         = H * D'; 
Ht         = D * H';
nan        = rho;

x          = sol(:);
x2         = x;
z          = sol(:);
d          = z;
% cache the factorization

for t = 1:MITR
    z_old   = z;

    alph    = 1/t;
%     alph    = 0.15;
    
    % x-update
    x1      = (1 - alph)*x2 + alph*x;
    gradx   = rho*(x  - B'*(z - d));
    grad    = Ht*(Hf*x1) - Hty;
    x       = x - (1/(2*nan))*(gradx + grad);
    x2      = (1 - alph)*x2 + alph*x;

    % z-update 
    Bx      = B*x;
    z       = shrinkage(Bx + d, lmb/rho); 

    % u-update
    tmp     = Bx - z;
    d       = d + tmp;
    
    if mod(t, 10) == 1
           
        s_norm   = norm(tmp);
        r_norm   = norm(-rho*(z - z_old));
               
        if s_norm > 10*r_norm
            rho  = rho*2;
            nan  = rho;
            d    = d/2;
        elseif r_norm > 10*s_norm
            rho  = rho/2;
            nan  = rho;
            d    = d*2;
        end  
        
        if PRNT
            fprintf('itr = %f, res = %f, dual = %f, rho = %f\n', t, r_norm, s_norm, rho); 
            temp = reshape(x, N, N, L);
            imagesc(temp(:,:,10)); colormap(gray)
            drawnow
        end
        
        if (s_norm < TOL) && (r_norm < TOL) 
            break;
        end   
    end
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );