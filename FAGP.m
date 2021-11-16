%% Fast approximate Gaussian Process (FAGP)

clear
% 
% [x1, x2, button] = ginput(30);
% % mouse left: space; 
% %       middle: countour; 
% %       right: obstacle.
% x = [x1'; x2'];
% y = button' - 2;

% x = [0.2, 0.4, 0.6, 0.4, 0.6, 0.9; ...
%      0.5, 0.5, 0.5,  0.8, 0.1, 0.6];
% y = [ -1,   0,   1,   -1,  -1,  -1];
x = [0.2, 0.4, 0.6, 0.6, 0.9, 0.55, 0.66, 0.66, 0.79, 0.08, 0.12, 0.29, 0.87, 0.91; ...
     0.5, 0.5, 0.5, 0.1, 0.6, 0.55, 0.54, 0.33, 0.58, 0.08, 0.30, 0.16, 0.12, 0.29];
% x = [0.2, 0.4, 0.6; ...
%      0, 0, 0]; % Aligned horizontally
% x = [0.5, 0.5, 0.5; ...
%      0.1, 0.5, 0.9]; % Aligned vertically
y = [ -1,   0,   1,  -1,  -1,    1,    1,    0,    0,   -1,   -1,   -1,   -1,   -1];
% x = [0.1, 0.5, 0.9]; % 1D case
% y = [-1, 0, 1];
% x = [0.35; 0.27];
% y = -1;

% Define the points under consideration
n = 5; % # of eigenvalues
% x = [0.2, 0.4, 0.6, 0.6, 0.9, 0.55, 0.66, 0.66, 0.79, 0.08, 0.12, 0.29, 0.87, 0.91; ...
%      0.5, 0.5, 0.5, 0.1, 0.6, 0.55, 0.54, 0.33, 0.58, 0.08, 0.30, 0.16, 0.12, 0.29];
% y = [ -1,   0,   1,  -1,  -1,    1,    1,    0,    0,   -1,   -1,   -1,   -1,   -1];

% X = [linspace(-1,1,50); linspace(-1,1,50)]';
[ X1, X2 ] = meshgrid(linspace(0,1,50), linspace(0,1,50));
X = [ X1(:), X2(:) ];

% Define the phi functions
% phi accepts row vector n and column vector x
%     it returns a length(x)-by-length(n) matrix

l = 1; % Scale factor
alpha = 1; % Global scale factor
epsilon = 1/(sqrt(2)*l); % Parameter depending on scale factor
R = 1;

for eigv = n:2:n+1
    tic
    [ K_tilde, K_app, Ks_app, indices ] = approximateKernel(x', X, eigv, epsilon, alpha);

    ys = K_tilde*y';
    toc

    figure
    hold on
    surface(X1, X2, reshape(ys, 50, 50) - max(ys), 'FaceColor','interp','EdgeColor','interp');
    % quiver(xs(1,:), xs(2,:), ys_grad(1,:), ys_grad(2,:),'color',[.2 .2 .2]);
    plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
    plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
    plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points
    contour(X1, X2, reshape(ys, 50, 50), [0,0], 'linewidth',2,'color',rand(1,3));
    title(['FAGP using ', num2str(eigv), ' eigenvalues'])
    hold off
end

% Classic GP
l = 1; % Scale factor
epsilon = 1/(sqrt(2)*l); % Parameter depending on scale factor
tic
K = exp(-epsilon^2*pdist2(x', x').^2);
Ks = exp(-epsilon^2*pdist2(X, x').^2);

% Thin plate cov
% K = 2*abs(pdist2(x', x').^3) - 3*R*pdist2(x', x').^2 + R^3;
% Ks = 2*abs(pdist2(X, x').^3) - 3*R*pdist2(X, x').^2 + R^3;

ys_std = Ks/K*y';
toc
% ys2 = Ks_app/K_app*y';

%% Plots

% figure
% hold on
% surface(X1, X2, reshape(ys, 50, 50) - max(ys), 'FaceColor','interp','EdgeColor','interp');
% plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
% plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
% plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points
% contour(X1, X2, reshape(ys, 50, 50), [0,0], 'linewidth',2,'color',rand(1,3));
        
figure
hold on
surface(X1, X2, reshape(ys_std, 50, 50) - max(ys_std), 'FaceColor','interp','EdgeColor','interp');
plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points
contour(X1, X2, reshape(ys_std, 50, 50), [0,0], 'linewidth',2,'color',rand(1,3));
title('Classic GP formula');

%% Approximated Gaussian Kernel

function [ K_tilde, K_approx, Ks_approx, idx_comb ] = approximateKernel(x, xp, n, ep, alpha)

    % Combinations
    [ index1, index2 ] = ndgrid(1:n, 1:n);
    idx_comb = [ index1(:), index2(:) ];
    phi_comb = zeros(size(x,1), size(idx_comb,1));
    phip_comb = zeros(size(xp,1), size(idx_comb,1));
    lambda_comb = zeros(1,size(idx_comb,1));
%     Lambda_hat = zeros(n^2);
%     K1_tilde = zeros(size(xp,1), size(x,1));
%     K2_tilde = zeros(size(x,1));
%     K_approx = zeros(size(x,1));
%     Ks_approx = zeros(size(xp,1), size(x,1));
%     tic
    for idx = 1:size(idx_comb,1)
        % phi_comb is phi_m(x1)*phi_p(x2), where x1, x2 the dims of x and
        % (m,p) all the grid combinations of n eigenvalues (n^dims in
        % total)
        phi_comb(:,idx) = eigenFnct(x(:,1), idx_comb(idx,1), ep, alpha).*...
            eigenFnct(x(:,2), idx_comb(idx,2), ep, alpha);
        lambda_comb(idx) = eigenValue(idx_comb(idx,1), ep, alpha)*eigenValue(idx_comb(idx,2), ep, alpha);
        
        phip_comb(:,idx) = eigenFnct(xp(:,1), idx_comb(idx,1), ep, alpha).*...
            eigenFnct(xp(:,2), idx_comb(idx,2), ep, alpha);
%         lambdap_comb(idx) = eigenValue(idx_comb(idx,1), ep, alpha)*eigenValue(idx_comb(idx,2), ep, alpha);
        
%         K_approx = K_approx + lambda_comb(idx)*phi_comb(:,idx_comb(idx,1))* ...
%             phi_comb(:,idx_comb(idx,2))'; % WRONG! I ALREADY DID THE COMBINATIONS OF PHI, AND
        % THE SUMMATION IS OVER THE SAME INDEX bold n!!!
        
%         Ks_approx = Ks_approx + lambda_comb(idx)*phip_comb(:,idx_comb(idx,1))* ...
%             phi_comb(:,idx_comb(idx,2))';
        
%         inv_SigmaN = 1./1e-5*eye(size(x,1));
%         Lambda_hat = Lambda_hat + phi_comb(:,idx_comb(idx,1))'* ...
%             inv_SigmaN*phi_comb(:,idx_comb(idx,2));
    end
%     toc
    inv_SigmaN = 1./1e-5*eye(size(x,1));
    Lambda_hat = phi_comb'*inv_SigmaN*phi_comb + diag(1./lambda_comb);
    
    K_tilde = phip_comb*diag(lambda_comb)*phi_comb'*(inv_SigmaN - inv_SigmaN*phi_comb/Lambda_hat*phi_comb'*inv_SigmaN);
    
%     for idx = 1:size(idx_comb,1)
%         K1_tilde = K1_tilde + phip_comb(:,idx_comb(idx,1))* ...
%             lambda_comb(idx)*phi_comb(:,idx_comb(idx,2))';
% %         K2_tilde = K2_tilde + phi_comb(:,idx_comb(idx,1))/ ...
% %             Lambda_hat(idx,idx)*phi_comb(:,idx_comb(idx,2))';
%     end
%     K2_tilde = K2_tilde + phi_comb(:,idx_comb(idx,1))/ ...
%     Lambda_hat(idx,idx)*phi_comb(:,idx_comb(idx,2))';
%     K_tilde = K1_tilde*(inv_SigmaN - inv_SigmaN*K2_tilde*inv_SigmaN);
    
    K_approx = phi_comb*diag(lambda_comb)*phi_comb';
    Ks_approx = phip_comb*diag(lambda_comb)*phi_comb';
%     for ii = 1:size(idx_comb,1)
%         K_approx = K_approx + lambda_comb(ii)*phi_comb(:,ii)*phi_comb(:,ii)';
%         Ks_approx = Ks_approx + lambda_comb(ii)*phip_comb(:,ii)*phi_comb(:,ii)';
%     end
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

