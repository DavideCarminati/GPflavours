%% KernelsSeries IN 2D!!
% This is an example of a kernel evaluated through series rather than
% through a closed form of the kernel
% The series is K(x,z) = sum lam_n phi_n(x) phi_n(z)
% The kernels of interest are the Gaussian kernels
clear

% Define the points under consideration
n = 10; % # of eigenvalues
X = [linspace(-1,1,50); linspace(-1,1,50)]';
[ X1, X2 ] = meshgrid(linspace(-1,1,50), linspace(-1,1,50));
% X = [ X1(:), X2(:) ];

l = 1; % Scale factor
alpha = 0.01; % Global scale factor
epsilon = 1/(sqrt(2)*l); % Parameter depending on scale factor


% Compute the n-length series approximation to the kernels
% This is the Phix*Lambda*Phiz' computation
K = exp(-epsilon^2*pdist2(X, X).^2);
[ K_approx, indices ] = approximateKernel(X, n, epsilon, alpha);

%% Approximated Gaussian Kernel

function [ K_approx, idx_comb ] = approximateKernel(x, n, ep, alpha)

    % Combinations
    [ index1, index2 ] = ndgrid(1:n, 1:n);
    idx_comb = [ index1(:), index2(:) ];
    phi_comb = zeros(size(x,1), size(idx_comb,1));
    lambda_comb = zeros(1,size(idx_comb,1));
    K_approx = zeros(size(x,1));
    for idx = 1:size(idx_comb,1)
        % phi_comb is phi_m(x1)*phi_p(x2), where x1, x2 the dims of x and
        % (m,p) all the grid combinations of n eigenvalues (n^dims in
        % total)
        phi_comb(:,idx) = eigenFnct(x(:,1), idx_comb(idx,1), ep, alpha).*...
            eigenFnct(x(:,2), idx_comb(idx,2), ep, alpha);
        lambda_comb(idx) = eigenValue(idx_comb(idx,1), ep, alpha)*eigenValue(idx_comb(idx,2), ep, alpha);
        
        K_approx = K_approx + lambda_comb(idx)*phi_comb(:,idx_comb(idx,1))* ...
            phi_comb(:,idx_comb(idx,2))';
        % Alternatively: K_approx = phi_comb*diag(lambda_comb)*phi_comb';
        
        % Generate a plot to check the shape of the first eigenfunctions
%         if idx <= 9
%             figure
%             [ X1, X2 ] = meshgrid(linspace(-1,1,50), linspace(-1,1,50));
%             surf(X1, X2, eigenFnct(x(:,1), idx_comb(idx,1), ep, alpha)*...
%                 eigenFnct(x(:,2), idx_comb(idx,2), ep, alpha)')
%             title(['Combination: ', num2str(idx_comb(idx,:))])
%         end
    end
end

function phi = eigenFnct(x_1D, n_eigv, ep, alpha)

    % Compute the eigenfunction phi corresponding to the eigenvalue n_eigv
    % using one of the dimensions of x "x_1D".
    beta = (1 + (2*ep/alpha)^2)^0.25;
    Gamma = sqrt(beta/(2^(n_eigv-1)*gamma(n_eigv)));
    delta2 = alpha^2/2*(beta^2 - 1);
    
    phi = Gamma*exp(-delta2*x_1D.^2).*hermiteH(n_eigv-1, alpha*beta*x_1D);
end

function lambda = eigenValue(n_eigv, ep, alpha)
    % Decreasing eigenvalues computation
    beta = (1 + (2*ep/alpha)^2)^0.25;
    delta2 = alpha^2/2*(beta^2 - 1);
    lambda = sqrt(alpha^2/(alpha^2 + delta2 + ep^2))*(ep^2/(alpha^2 + delta2 + ep^2))^(n_eigv-1);
end