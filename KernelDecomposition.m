%% KernelsSeries IN 2D!!
% This is an example of a kernel evaluated through series rather than
% through a closed form of the kernel
% The series is K(x,z) = sum lam_n phi_n(x) phi_n(z)
% The kernels of interest are the Gaussian kernels


% Define the points under consideration
n = 20; % # of eigenvalues
X = [linspace(-1,1,50); linspace(-1,1,50)]';
[ X1, X2 ] = meshgrid(linspace(-1,1,50), linspace(-1,1,50));
% X = [ X1(:), X2(:) ];

% Define the phi functions
% phi accepts row vector n and column vector x
%     it returns a length(x)-by-length(n) matrix

l = 0.1; % Scale factor
alpha = 1; % Global scale factor
epsilon = 1/(sqrt(2)*l); % Parameter depending on scale factor
Phi = zeros(size(X,1), n); % (num data points)x(num eigenvalues)
lam = zeros(1, n);
for ii = 1:n
    [ Phi(:,ii), lam(ii) ] = basisFun(X, ii, epsilon, alpha);
end

% Compute the n-length series approximation to the kernels
% This is the Phix*Lambda*Phiz' computation

% Kse = bsxfun(@times,Phi,lam)*Phi'; % Equivalent formulation
Kse = Phi*diag(lam)*Phi';

%% Approximated Gaussian Kernel

function [ phi, lambda ] = basisFun(x, n, ep, alpha)

    % Parameters
    beta = (1 + (2*ep/alpha)^2)^0.25;
    Gamma = sqrt(beta/(2^(n-1)*gamma(n)));
    delta2 = alpha^2/2*(beta^2 - 1);
    
    out1D = Gamma*exp(-delta2*x(:,1).^2).*hermiteH(n-1, alpha*beta*x(:,1)); % First dimension of x
    out2D = Gamma*exp(-delta2*x(:,2).^2).*hermiteH(n-1, alpha*beta*x(:,2)); % Second dimension of x
    phi = out1D.*out2D;
    
    % Decreasing eigenvalues computation
    lambda = sqrt(alpha^2/(alpha^2 + delta2 + ep^2))*(ep^2/(alpha^2 + delta2 + ep^2))^(n-1);
    lambda = lambda^2; % Eig multiplication since we are in 2D
    
    % Checking the shape of the eigenfunctions for the n eigenvalue
    if n == 1
        figure
        [ x1, x2 ] = meshgrid(x(:,1), x(:,2));
        surf(x1, x2, out1D*out2D');
    end
    
end


