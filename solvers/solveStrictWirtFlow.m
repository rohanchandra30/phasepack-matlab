
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

