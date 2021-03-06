%                           initOpticalSpectral.m
%
%  Initializer recently proposed based on an optimal spectral method. The
%  plain vanilla spectral initializer computes the largest eigenvector of Y
%  = 1/m sum(yi * ai * ai') for i = 1 to m.  The truncated version of this
%  method throws away some of the rows.  The optimal spectral nitializer
%  computes the largest eigenvector of Y = 1/m sum(T(yi) * ai * ai') for i
%  = 1 to m, where T(yi) is a function of yi given in equation (5) of the
%  paper cited below.
%
%% I/O
%  Inputs:
%     A:  m x n matrix (or optionally a function handle to a method) that
%         returns A*x.
%     At: The adjoint (transpose) of 'A'. If 'A' is a function handle, 'At'
%         must be provided.
%     b0: m x 1 real,non-negative vector consists of all the measurements.
%     n:  The size of the unknown signal. It must be provided if A is a 
%         function handle.
%     isTruncated (boolean): If true, use the 'truncated' initializer that
%                            uses a sub-sample of the measurement.
%     isScaled (boolean):    If true, use a least-squares method to
%                            determine  the optimal scale of the
%                            initializer.
%
%     Note: When a function handle is used, the value of 'n' (the length of
%     the unknown signal) and 'At' (a function handle for the adjoint of
%     'A') must be supplied.  When 'A' is numeric, the values of 'At' and
%     'n' are ignored and inferred from the arguments.
%
%  Outputs:
%     x0:  A n x 1 vector. It is the guess generated by the spectral method
%          for  the unknown signal.
%
%  See the script 'testInitOptimalSpectral.m' for an example of proper usage of
%  this function.
%
%% Notation
%  Our notation follows the TWF paper.
%  ai is the conjugate transpose of the ith row of A.
%  yi is the ith element of y, which is the element-wise square of the
%  measurements b0.
%
%% Algorithm Description.
%  Calculate the leading eigenvector of a matrix Y, where Y = 1/m sum(T(yi)
%  * ai * ai') for i = 1 to m, where T() is a "pre-processing" function. 
%  The method return this leading eigenvector,
%  which is calculated using Matlab's eigs() routine. 
%
%  Note: This implementation differs from the paper in several ways that
%  make it more efficient and robust.
%  The papers below recommend using the power method to compute the leading
%  eigenvector.  Our implemention
%  uses Matlab's built-in function eigs() to get the leading eigenvector
%  because of greater efficiency.
%
%  Also, the authors define the pre-processing function
%                 T(z) = (z-1)/(z+sqrt(delta)-1), 
% where delta is the ratio of number of measurements to number of 
% dimensions.  This formula assumes that the measurements are Gaussian with
% variance 1/n, and the unknown signal has length sqrt(n).  This assumption
% is clearly violated by most real sensing matrices and signals.  However,
% note that this measurements model yields measurements y = abs(Ax)^2 that
% have expected value E(y)=1.  For this reason, we normalize the
% measurements to have mean 1 before we apply the pre-processing function.
% We then multiply the mean back into the results when we're done
% pre-processing.

%% References
%  Title:   Fundamental Limits of Weak Recovery with Applications to Phase 
%           Retrieval 
%  Place:   Equations (4) and (5) 
%  Authors: Marco Mondelli, Andrea Montanari 
%  Arxiv Address: https://arxiv.org/pdf/1708.05932.pdf
%

%
% PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
% Christoph Studer, & Tom Goldstein 
% Copyright (c) University of Maryland, 2017

%% -----------------------------START----------------------------------


function [x0] = initOptimalSpectral(A,At,b0,n,isScaled,verbose)

% If A is a matrix, infer n and At from A. Then, transform matrix into 
% a function handle.
if isnumeric(A)
    n = size(A, 2);
    At = @(x) A' * x;
    A = @(x) A * x;
end

m = numel(b0);                % number of measurements

if ~exist('verbose','var') || verbose
fprintf(['Estimating signal of length %d using a optical spectral ',...
         'initializer with %d measurements...\n'],n,m);
end

% Measurements as defined in the paper
y = b0.^2;                             
delta = m/n;    % Used in equation (5) of paper

% Normalize the measurements
ymean = mean(y);
y = y/ymean;

% Apply pre-processing function
T = (y-1)./(1+sqrt(delta)-1);

% Un-normalize the measurements
y = y*ymean;

% Build the function handle associated to the matrix Y
Yfunc = @(x) 1/m*At(T.*A(x));

% Our implemention uses Matlab's built-in function eigs() to get the leading
% eigenvector because of greater efficiency.
% Create opts struct for eigs
opts = struct;
opts.isreal = false;

% Get the eigenvector that corresponds to the largest eigenvalue of the
% associated matrix of Yfunc.
[x0,~] = eigs(Yfunc, n, 1, 'lr', opts);

% This part does not appear in the Null paper. We add it for better
% performance. Rescale the solution to have approximately the correct
% magnitude
if isScaled
    b = b0;
    Ax = abs(A(x0));
    
    % solve min_s || s|Ax| - b ||
    u = Ax.*b;
    l = Ax.*Ax;
    s = norm(u(:))/norm(l(:));
    x0 = x0*s;                   % Rescale the estimation of x
end

if ~exist('verbose','var') || verbose
    fprintf('Initialization finished.\n');
end

end
