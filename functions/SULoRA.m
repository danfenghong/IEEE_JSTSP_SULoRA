function [scaled_X, theta] = SULoRA(Y, A0, X, param, maxiter, theta)

alfa = param.alfa;
beta = param.beta;
gama = param.gama;
 
A = A0;
epsilon = 1e-6;
iter = 0;

[D, N] = size(Y);
G = zeros(D, D);
H = zeros(size(X));
J = zeros(size(X));

lamda1 = zeros(size(G));
lamda2 = zeros(size(X));
lamda3 = zeros(size(X));

stop = false;
mu = 1e-3;
rho = 1.5;
mu_bar = 1E+6;

while ~stop && iter < maxiter+1
    
    iter = iter + 1;
    
    % update M
    theta = (alfa * (Y * Y') + mu * G + lamda1) / (alfa * (Y * Y') + (Y - A * X) * (Y - A * X)' + mu * eye(size(Y * Y')));
    
    % update X
    X = ((theta * A)' * (theta * A) + 2 * mu * eye(size((theta * A)' * (theta * A)))) \ ((theta * A)' * theta * Y ...
         + mu * H + lamda2 + mu * J + lamda3); 
    
    % update G by solving the low-rank problem
    Resi_G = theta - lamda1 / mu;
    [U1, S1, V1] = svd(Resi_G, 'econ');
    diagS = diag(S1);
    svp = length(find(diagS > beta / mu));
    if svp >= 1
        diagS = diagS(1 : svp) - beta / mu;
    else
        svp = 1;
        diagS = 0;
    end
    G = U1(:, 1 : svp) * diag(diagS) * V1(:, 1:svp)'; 
    
    % update H by soft threshold
    H = max(abs(X - lamda2 / mu) - (gama / mu), 0).* sign(X - lamda2 / mu); 
    
    % update J
    J = max(X - lamda3 / mu, 0);  
   
    % update lamda1-3
    lamda1 = lamda1 + mu * (G - theta);
    lamda2 = lamda2 + mu * (H - X);
    lamda3 = lamda3 + mu * (J - X);
    mu = min(mu * rho, mu_bar);
    
    % check convergence
    r_G = norm(G-theta,'fro');
    r_H = norm(H-X,'fro');
    r_J = norm(J-X,'fro');
    
    if r_H < epsilon && r_G < epsilon && r_J < epsilon
        stop = true;
        break;
    end
    
end

scaled_X = X./repmat(sum(X), size(X,1), 1); % scaling the abundances
end
