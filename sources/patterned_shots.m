function [shot_data, shots, num_filters, cca, filter_type, filter_set] =...
    patterned_shots(I, compression_ratio, filtertype)
% patterned_shots returns the compressive measurements of the spectral
% image I under the patterned architecture for a specific compression
% ratio. This function also returns the number of snapshots (shots)
% required for obtaining the compressive measurements as well as the 3D
% models of the coded apertures (cca) used in the acquisition of the
% compressive measurements.
%
% [shot_data, shots, num_filters, cca, filter_type] =...
%    patterned_shots(I, compression_ratio, filtertype)
%
%   Inputs:
%   I                   = input spectral image
%   compression_ratio   = compressio ratio (rho)
%   filtertype
%
%   'binary':Selects a set of ideal bandpass optical filters covering all
%   wavelength spectrum.
%
%   'bandpass':Selects a set of real bandpass optical filters covering all
%   wavelength spectrum. Each filter response is obtained using the fir1
%   matlab function.
%
%   'random-uniform':Each filter is obtained as a random vector whose
%   entries are i.i.d. random samples following a uniform distribution.
%   The sum of the filter magnitudes along the set of filters in a single
%   spectral band is normalized to one. 
%
%   Outputs: 
%   shot_data   = compressive measurements
%   shots       = number of required shots
%   cca         = collection of coded aperture binary models
%   filter_type = 
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


if (nargin == 2)
    filtertype = 'bandpass';
end

[N1, N2, L] = size(I);
num_filters = floor(L * compression_ratio);
shots       = floor(L * compression_ratio);


[cca, filter_type,filter_set]  = colored_ca(shots, N1, N2, L, num_filters, filtertype);
shot_data = zeros(N1, N2, shots);
for i = 1 : shots
    shot_data(:,:,i) = sum(I .* cca{i},3);
end

% shot_data = shot_data/max(shot_data(:));