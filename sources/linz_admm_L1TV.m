function x = linz_admm_L1TV(H, Psi, Lo, y, tau1, tau2, parm, sol)
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
[N, ~, L]   = size(sol);
Hty         = H'*y;
% Ltl         = Lo'*Lo;
% Psitpsi     = speye(N*N*L);
% Btb         = (H'*H + rho*(Ltl + Psitpsi));

nan        = rho;

x          = sol(:);
x2         = x;

z1          = Lo * sol(:);
d1          = z1;
z2          = sol(:);
d2          = z2;
% cache the factorization

for t = 1:MITR
    z_old1   = z1;
    z_old2   = z2;
    
%     % x-update
%     grad1   = Lo' * (z1 - d1); 
%     grad2   = Psi' * (z2 - d2);
%     rl      = Hty + rho * (grad1 + grad2);
%     x       = cgsolve(Btb, rl);

%     alph    = 1/t;
    alph    = 0.15;
    
    % x-update
    x1      = (1 - alph)*x2 + alph*x;
    gradx2  = rho*(x  - Psi'*(z2 - d2));
    gradx1  = rho*(Lo'*(Lo*x  - z1 + d1));
    grad    = H'*(H*x1) - Hty;
    x       = x - (1/(2*nan))*(gradx1 + gradx2 + grad);
    x2      = (1 - alph)*x2 + alph*x;
    
    % z-update 
    Lx      = Lo * x;
    z1      = shrinkage(Lx + d1, tau1/rho); 
    Bx      = Psi * x;
    z2      = shrinkage(Bx + d2, tau2/rho); 

    % u-update
    tmp1     = Lx - z1;
    d1       = d1 + tmp1;
    tmp2     = Bx - z2;
    d2       = d2 + tmp2;
    
    if mod(t, 10) == 1
           
        s_norm   = sqrt(norm(tmp1)^2 + norm(tmp2)^2);
        r_norm   = norm(-rho*(z1 - z_old1)) + norm(-rho*(z2 - z_old2));
               
        if s_norm > 10*r_norm
            rho  = rho*2;
            nan  = rho;
            d1    = d1/2;
        elseif r_norm > 10*s_norm
            rho  = rho/2;
            nan  = rho;
            d1    = d1*2;
        end  
        
        if PRNT
            fprintf('itr = %f, res = %f, dual = %f, rho = %f\n', t, r_norm, s_norm, rho); 
%             temp = reshape(x, N, N, L);
%             imagesc(temp(:,:,10)); colormap(gray)
%             drawnow
        end
        
        if (s_norm < TOL) && (r_norm < TOL) 
            break;
        end   
    end
end


function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
    
    % Hector gave me this code
% function [x, res, iter] = cgsolve(A, b)
% 
% tol = 1e-6; 
% maxiter = 10; 
% 
% implicit = isa(A,'function_handle');
% 
% x = zeros(length(b),1);
% r = b;
% d = r;
% delta = r'*r;
% delta0 = b'*b;
% numiter = 0;
% bestx = x;
% bestres = sqrt(delta/delta0); 
% while ((numiter < maxiter) && (delta > tol^2*delta0))
%     
% 
%   % q = A*d
%   if (implicit), q = A(d);  else,  q = A*d;  end
% 
%   alpha = delta/(d'*q);
%   x = x + alpha*d;
% 
%   if (mod(numiter+1,50) == 0)
%     % r = b - Aux*x
%     if (implicit), r = b - A(x);  else,  r = b - A*x;  end
%   else
%     r = r - alpha*q;
%   end
% 
%   deltaold = delta;
%   delta = r'*r;
%   beta = delta/deltaold;
%   d = r + beta*d;
%   numiter = numiter+1;
% %   disp(numiter)
%   if (sqrt(delta/delta0) < bestres)
%     bestx = x;
%     bestres = sqrt(delta/delta0);
%   end    
% 
% end
% 
% x = bestx;
% res = bestres;
% iter = numiter;