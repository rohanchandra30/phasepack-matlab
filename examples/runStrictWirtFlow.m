%% -----------------------------START----------------------------------- 
 

clear;
close all;

n = 100;          % Dimension of unknown vector
m = 5 * n;        % Number of measurements
isComplex = true; % If the signal and measurements are complex

%%  Build a random test problem
fprintf('Building test problem...\n');
[A, xt, b0] = buildTestProblem(m, n, isComplex);

% Options
opts = struct;
opts.initMethod = 'spectral';
opts.algorithm = 'strictwirtflow';
opts.isComplex = isComplex;
opts.tol = 1e-10;
opts.verbose = 2;
opts.maxIters = 10000;
opts.xt = xt;
%% Try to recover x
fprintf('Running algorithm...\n');
[x, outs, opts] = solvePhaseRetrieval(A, A', b0, n, opts);

%% Determine the optimal phase rotation so that the recovered solution
%  matches the true solution as well as possible.  
alpha = (x'*xt)/(x'*x);
x = alpha * x;

%% Determine the relative reconstruction error.  If the true signal was 
%  recovered, the error should be very small - on the order of the numerical
%  accuracy of the solver.
reconError = norm(xt-x)/norm(xt);
fprintf('relative recon error = %d\n', reconError);

% Plot a graph of error(definition depends on if opts.xt is provided) versus
% the number of iterations.
plotErrorConvergence(outs, opts)

% Plot a graph of the recovered signal x against the true signal xt.
plotRecoveredVSOriginal(x,xt);