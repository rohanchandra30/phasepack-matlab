%                           solveStrictWirtFlow.m
%
%  Strict Implementation of the Wirtinger Flow (WF) algorithm proposed in
%  the paper. This code mirrors the experiment performed on random 1-D
%  gaussian data in section 4.2 of the paper.
%
%% I/O
%  Inputs:
%     A:    m x n matrix or a function handle to a method that
%           returns A*x.     
%     At:   The adjoint (transpose) of 'A'. If 'A' is a function handle,
%           'At' must be provided.
%     b0:   m x 1 real,non-negative vector consists of all the measurements.
%     x0:   n x 1 vector. It is the initial guess of the unknown signal x.
%     opts: A struct consists of the options for the algorithm. For details,
%           see header in solvePhaseRetrieval.m or the User Guide.
%
%     Note: When a function handle is used, the
%     value of 'At' (a function handle for the adjoint of 'A') must be 
%     supplied.
% 
%  Outputs:
%     sol:  n x 1 vector. It is the estimated signal.
%     outs: A struct consists of the convergence info. For details,
%           see header in solvePhaseRetrieval.m or the User Guide.
%  
%  
%  See the script 'testWirtFlow.m' for an example of proper usage of this 
%  function.
%
%% Notations
%  x is the estimation of the signal. y is the vector of measurements such
%  that yi = |<ai,x>|^2 for i = 1,...,m
%
%% Algorithm Description
%  WF successively refines the estimate via an update rule that bears a
%  strong resemblance to a gradient descent scheme. Specifically, at each
%  iteration, x = x + mu/m * gradient log-likelihood of x given y For the
%  detailed formulation of "gradient log-likelihood of x given y" and a
%  detailed explanation of the theory, see the WF paper referenced below.
%  
%% References
%  Paper Title:   Phase Retrieval via Wirtinger Flow: Theory and Algorithms
%  Place:         Chapter 4.2
%  Authors:       Emmanuel Candes, Xiaodong Li, Mahdi Soltanolkotabi
%  arXiv Address: https://arxiv.org/abs/1407.1065
%  
% PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
% Christoph Studer, & Tom Goldstein 
% Copyright (c) University of Maryland, 2017

%% -----------------------------START-----------------------------------

function [x, outs] = solveStrictWirtFlow(A, At, b0, x0, opts)

% Functions to compute objective and gradient
f = @(z) 0.5 * norm(abs(z).^2 - b0.^2)^2;
gradf = @(z) 1/length(b0) * (abs(z).^2 - b0.^2) .* z;

% Step size rule as given in section 4.2 of https://arxiv.org/pdf/1407.1065.pdf
tau0 = 330;
mu = @(t) min(1-exp(-t/tau0), 0.2);

% initialize
residuals = [];
normest = sqrt(sum(b0.^2)/numel(b0.^2)); % Estimate norm to scale eigenvector  
x_old = normest*x0; % Apply Scaling


% Start Gradient Descent
for i=1:opts.maxIters
        
    % Gradient step
    x_new = x_old - mu(i)/normest^2 *At(gradf(A(x_old)));
    
    error = norm(opts.xt - exp(-1i*angle(trace(opts.xt'*x_new))) * x_new, 'fro')/norm(x_new,'fro');
    if error <= opts.tol
        break
    end
    
    x_old = x_new;
    
    % Record residuals for plottign convergance
    residuals(i) = norm(At(A(x_new)-b0.*sign(A(x_new))))/norm(b0.*sign(A(x_new)));
    
end
x = x_new;
outs = struct;
outs.residuals = residuals;

end

