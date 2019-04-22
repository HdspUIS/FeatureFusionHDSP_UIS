function [I_HS] = spatial_blurring(I, p, window, sigma, dec_type)
% spatial_blurring returns a blurred and downsampled version of the input
% spectral image I. I_HS is obtained by applying a Gaussian filter to each
% spectral band of the input spectral image, where each blurred band is
% downsampled according to the decimation ratio p. The Gaussian filter
% parameters are given by the standard deviation sigma and the window size.
%
% [I_HS] = spatial_blurring(I, p, window, sigma)
%
%   Inputs:
%   I               = input spectral image
%   p               = spatial decimation ratio
%   window          = window size
%   sigma           = standard deviation of the Gaussian filter
%   
%   Outputs:
%   I_HS            = blurred and downsampled version of the input image
%
%   Reference: 
%
%   [1] Juan Marcos Ramirez and Henry Arguello, "Spectral Image
%   Classification From Data Fusion Compressive Measurements"
%
%   Authors:
%   Juan Marcos Ramirez.
%   Universidad Industrial de Santander, Bucaramanga, Colombia
%   email: juanmarcos26@gmail.com
%
%   Date:
%   May, 2018
%
%   Copyright 2018 Juan Marcos Ramirez Rondon.  [juanmarcos26-at-gmail.com]

%   This program is free software; you can redistribute it and/or modify it
%   under the terms of the GNU General Public License as published by the
%   Free Software Foundation; either version 2 of the License, or (at your
%   option) any later version.
% 
%   This program is distributed in the hope that it will be useful, but
%   WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%   General Public License for more details.
% 
%   You should have received a copy of the GNU General Public License along
%   with this program; if not, write to the Free Software Foundation, Inc.,
%   675 Mass Ave, Cambridge, MA 02139, USA.

if (nargin == 4)
    dec_type = 'Gaussian_filter';
end

[N1, N2, L] = size(I);

if strcmp(dec_type,'Gaussian_filter')   
    kernel = gaussian_kernel(window, sigma);
    for i = 1:L
        I_temp = imfilter(I(:,:,i), kernel,'replicate');
        I_HS(:,:,i)   = I_temp(1:p:N1,1:p:N2);
    end
elseif strcmp(dec_type,'average')
    for i = 1 : N1/p
        for j = 1 : N2/p
            I_HS(i,j,:) = mean(reshape(I((i-1)*p + 1:i*p,(j-1)*p+1:j*p,:),[p^2 L]));
        end
    end
elseif strcmp(dec_type,'sum')
    for i = 1 : N1/p
        for j = 1 : N2/p
            I_HS(i,j,:) = sum(reshape(I((i-1)*p + 1:i*p,(j-1)*p+1:j*p,:),[p^2 L]));
        end
    end
end

end

function kernel = gaussian_kernel(window_size, sigma)
    x = -floor(window_size/2):floor(window_size/2);
    [X,Y] = meshgrid(x);    
    kernel = 1/(2*pi*(sigma^2)) * exp(-(X.^2 + Y.^2)/(2*(sigma^2)));
    kernel = kernel / sum(kernel(:));
end