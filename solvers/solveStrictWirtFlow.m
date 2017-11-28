
%% -----------------------------START-----------------------------------

function [x, outs] = solveStrictWirtFlow(A, At, b0, x0, opts)

% Functions to compute objective and gradient
f = @(z) 0.5 * norm(abs(z).^2 - b0.^2)^2;
gradf = @(z) (abs(z).^2 - b0.^2) .* z;

% initialize
x_old = x0;
t_0 = 330;
residuals = [];

% Start Gradient Descent
for i=1:opts.maxIters
    
    % Step size rule as given in section 4.2 of https://arxiv.org/pdf/1407.1065.pdf
    mu = min(1-exp((-1*i)/t_0), 0.2);
    
    % Gradient step
    x_new = x_old - mu*At(gradf(A(x_old)));
    
    error = norm(x_new - opts.xt)/norm(x_new);
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

