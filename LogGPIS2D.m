%% 2D case
% In 2D, I need to specify a Y vector of (dim+1)N points, since I have to
% specify a 2D gradient this time! Also, the kernel derivatives are vectors
% if referred to a single point.
clear
noise = 1e-3;
numTest = 40;
wallPoints = 10;
lambda = 50;
R = sqrt(5)/lambda; % Kernels length scale

% Test points
[ X1, X2 ] = meshgrid(linspace(0,5,numTest), linspace(0,5,numTest));
X = [ X1(:), X2(:) ];

% Placing obstacles and/or walls
circleRadius = 0.5;
x = [2.5, 2.5] + circleRadius * [cos(-pi:pi/4:(pi-1e-3))', sin(-pi:pi/4:(pi-1e-3))'];
x2 = [4, 1] + 1.25*circleRadius * [cos(-pi:pi/4:(pi-1e-3))', sin(-pi:pi/4:(pi-1e-3))'];

% x = [ x; x2 ];

% Square room with circle obstacle:
% x = [ x;
%       linspace(0.6, 4.4, wallPoints)', 0.5*ones(wallPoints,1);
%       linspace(0.6, 4.4, wallPoints)', 4.5*ones(wallPoints,1);
%       0.5*ones(wallPoints,1), linspace(0.6, 4.4, wallPoints)';
%       4.5*ones(wallPoints,1), linspace(0.6, 4.4, wallPoints)' ]; % working well with rbf kernel

% x = [ x;
%       0, 0 ];

y = zeros(size(x, 1), 1);

% y(end) = sqrt(2)*2;

% dy = [ 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2; -1; -sqrt(2)/2; % Gradient along x1
%         -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2 ]; % Gradient along x2

% dy = [ 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2; -1; -sqrt(2)/2; % Gradient along x1
%         1; sqrt(2)/2; 0; -sqrt(2)/2; -1; -sqrt(2)/2; 0; sqrt(2)/2 ]; % Gradient along x2

% dy = [ -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2;
%         0; -sqrt(2)/2; -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2 ];

dy = -[ -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2;
        0; -sqrt(2)/2; -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2 ]; % Normal to obstacle border
    
% dy = -[ -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2; -sqrt(2)/2;
%         0; -sqrt(2)/2; -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2; -sqrt(2)/2 ]; % Normal to obstacle border

% dy = [ dy(1:length(dy)/2); dy(1:length(dy)/2); dy(length(dy)/2+1:end); dy(length(dy)/2+1:end) ];

% Square room with circle obstacle - normal to obst
% dy = [ 1; sqrt(2)/2; 0; -sqrt(2)/2; -1; -sqrt(2)/2; 0; sqrt(2)/2; zeros(2*wallPoints,1); -ones(wallPoints,1); ones(wallPoints,1);
%         0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2; -1; -sqrt(2)/2; -ones(wallPoints,1); ones(wallPoints,1); zeros(2*wallPoints,1) ]; 

% Kernel selection

% TP kernel gives a wrong distance representation
% [ K_tilde, K ] = kernelFnct2D(x, x, R, 'ThinPlate');
% [ Ks_tilde, Ks ] = kernelFnct2D(X, x, R, 'ThinPlate');
% [ Kss_tilde, Kss ] = kernelFnct2D(X, X, R, 'ThinPlate');

% RBF kernel overestimates the distance
% [ K_tilde, K ] = kernelFnct2D(x, x, R, 'RBF');
% [ Ks_tilde, Ks ] = kernelFnct2D(X, x, R, 'RBF');
% [ Kss_tilde, Kss ] = kernelFnct2D(X, X, R, 'RBF');

% Matérn 5/2 kernel is the best and is two-times differentiable
[ K_tilde, K ] = kernelFnct2D(x, x, R, 'Matern');
[ Ks_tilde, Ks ] = kernelFnct2D(X, x, R, 'Matern');
[ Kss_tilde, Kss ] = kernelFnct2D(X, X, R, 'Matern');

% Log GPIS
y = exp(-y*lambda);% + noise*randn(size(x, 1), 1);
mu_g = Ks_tilde / K_tilde * [y; dy];
cov = Kss_tilde - Ks_tilde/K_tilde*Ks_tilde';

% GP without gradient information
mu2 = Ks/K*y;
dist2 = -(1 / lambda) * real(log((mu2)));

% recover the mean according to Log-GPIS
dist = -(1 / lambda) * real(log((mu_g(1:numTest^2))));
% grad = -mu_g(numTest^2+1:end)./(lambda*mu_g(1:numTest^2));
% I need to repeat elements of mu_g (ypred) 2 by 2, so that the total
% length is doubled to match grad length.
doublingIndex = floor(1:0.5:numTest^2+0.5); % creating index vector looking like [ 1 1 2 2 3 3 ...]
% grad_normalization = mu_g(1:numTest^2);
% grad_normalization = grad_normalization(doublingIndex);
% grad = -mu_g(numTest^2+1:end)./(lambda*grad_normalization);
% 
% grad = -mu_g(numTest^2+1:end);

normlz = sqrt(mu_g(numTest^2+1:2*numTest^2).^2 + mu_g(2*numTest^2+1:end).^2);
grad = -mu_g(numTest^2+1:end)./(normlz(doublingIndex)*lambda);

% dist_cov = 1./(lambda*mu_g(1:numTest^2))'*cov(1:numTest^2,1:numTest^2)*(1./(lambda*mu_g(1:numTest^2)));
dist_cov = (1./(lambda*mu_g(1:numTest^2))).^2.*cov(1:numTest^2,1:numTest^2);
% dist_cov = cov(1:numTest^2,1:numTest^2);

% Rotate the gradients so that they are normal to surfaces
% theta = -pi/2;
% rot = [ cos(theta)*eye(numTest^2), sin(theta)*eye(numTest^2); ...
%         -sin(theta)*eye(numTest^2), cos(theta)*eye(numTest^2) ];
% grad = rot*grad;

figure
hsurf = surface(X1, X2, reshape(dist, numTest, numTest) - 0*max(dist), 'FaceColor','interp','EdgeColor','interp');
% colormap gray;
hold on
plot(x(:,1), x(:,2), '.','markersize',28,'color',[.7 0.3 0]); %Interior points
quiver(x(:,1), x(:,2), dy(1:length(y)), dy(length(y)+1:end), 0.5 , 'g');
contour(X1, X2, reshape(dist, numTest, numTest), [0,0], 'w');
% quiver(X(:,1), X(:,2), grad(1:numTest^2), grad(numTest^2+1:end), 'w');
hsurf.Annotation.LegendInformation.IconDisplayStyle = 'off';
legend('Obstacle border', 'Normal to border')
xlabel('x_1 [m]')
ylabel('x_2 [m]')
% xlim([0, 5])
% ylim([0, 5])
colorbar%('Ticks', [min(diag(ysd))-max(diag(ysd)), 0])
title('Predictive mean')
% view(3)

figure
hsurf_cov = surface(X1, X2, reshape(min(10,diag(dist_cov)), numTest, numTest) - 0*max(min(100,diag(dist_cov))), 'FaceColor','interp','EdgeColor','interp');
% colormap gray;
hold on
plot(x(:,1), x(:,2), '.','markersize',28,'color',[.7 0.3 0]); %Interior points
quiver(x(:,1), x(:,2), dy(1:length(y)), dy(length(y)+1:end), 0.5 , 'g');
contour(X1, X2, reshape(dist, numTest, numTest), [0,0], 'w');
% quiver(X(:,1), X(:,2), grad(1:numTest^2), grad(numTest^2+1:end), 'w');
hsurf_cov.Annotation.LegendInformation.IconDisplayStyle = 'off';
legend('Obstacle border', 'Normal to border')
xlabel('x_1 [m]')
ylabel('x_2 [m]')
% xlim([0, 5])
% ylim([0, 5])
colorbar%('Ticks', [min(diag(ysd))-max(diag(ysd)), 0])
title('Predictive covariance')

figure
surface(X1, X2, reshape((mu_g(1:numTest^2)), numTest, numTest), 'FaceColor','interp','EdgeColor','interp');
hold on
plot(x(:,1), x(:,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
xlabel('x_1 [m]')
ylabel('x_2 [m]')
title('Classic GP before log transform');

figure
surface(X1, X2, reshape(dist2, numTest, numTest), 'FaceColor','interp','EdgeColor','interp')
title('GP without gradient information')

figure
dist_1d = diag(reshape(dist, numTest, numTest));
plot(linspace(0, 5, numTest), dist_1d)
title('Distance from 0,0 to 5,5')

function [ K_tilde, K ] = kernelFnct2D(x1, x2, R, Kerneltype)

    N = size(x1,1);
    M = size(x2,1);
    % Thin plate covariance function
    if strcmp(Kerneltype, 'ThinPlate')
        K = 2*abs(pdist2(x1, x2).^3) - 3*R*pdist2(x1, x2).^2 + R^3; % Basis function
        % How to differenciate in nD kernels: 
        % https://math.stackexchange.com/questions/84331/does-this-derivation-on-differentiating-the-euclidean-norm-make-sense

        dKx1 = [ 6*pairwiseDiff(x1(:,1), x2(:,1)).*(pdist2(x1, x2) - R); 
                 6*pairwiseDiff(x1(:,2), x2(:,2)).*(pdist2(x1, x2) - R) ];
    %     dKx2 = -dKx1'; % For stationary kernels holds this property, but note
    %     that the transposition of a marix containing matrices is the
    %     transpose of submatrices inside the outer transposed matrix
        dKx2 = [ (6*pairwiseDiff(x2(:,1), x1(:,1)).*(pdist2(x2, x1) - R))', ...
                 (6*pairwiseDiff(x2(:,2), x1(:,2)).*(pdist2(x2, x1) - R))' ];

        dKx2 = [ -(6*pairwiseDiff(x1(:,1), x2(:,1)).*(pdist2(x1, x2) - R)), ...
                 -(6*pairwiseDiff(x1(:,2), x2(:,2)).*(pdist2(x1, x2) - R)) ];

        ddK = [ -6*pairwiseDiff(x1(:,1), x2(:,1)).*pairwiseDiff(x1(:,1), x2(:,1))./(pdist2(x1, x2)+1e-10) + pdist2(x1, x2) - R, ...
                -6*pairwiseDiff(x1(:,1), x2(:,1)).*pairwiseDiff(x1(:,2), x2(:,2))./(pdist2(x1, x2)+1e-10);
                -6*pairwiseDiff(x1(:,2), x2(:,2)).*pairwiseDiff(x1(:,1), x2(:,1))./(pdist2(x1, x2)+1e-10), ...
                -6*pairwiseDiff(x1(:,2), x2(:,2)).*pairwiseDiff(x1(:,2), x2(:,2))./(pdist2(x1, x2)+1e-10) + pdist2(x1, x2) - R ];
        K_tilde = [ K, dKx2; dKx1, ddK ];
    elseif strcmp(Kerneltype, 'RBF')
        sigma = 1;
        K = sigma*exp(-pdist2(x1, x2)/(2*R^2));
        dKx1 = [ -1/R^2*pairwiseDiff(x1(:,1), x2(:,1)).*K;
                 -1/R^2*pairwiseDiff(x1(:,2), x2(:,2)).*K ];
        dKx2 = [ 1/R^2*pairwiseDiff(x1(:,1), x2(:,1)).*K, ...
                 1/R^2*pairwiseDiff(x1(:,2), x2(:,2)).*K ];
        ddK = [ (1/R^2 - 1/R^4*pairwiseDiff(x1(:,1), x2(:,1)).*pairwiseDiff(x1(:,1), x2(:,1))).*K, ...
                (-1/R^4*pairwiseDiff(x1(:,1), x2(:,1)).*pairwiseDiff(x1(:,2), x2(:,2))).*K; 
                (-1/R^4*pairwiseDiff(x1(:,2), x2(:,2)).*pairwiseDiff(x1(:,1), x2(:,1))).*K, ...
                (1/R^2 - 1/R^4*pairwiseDiff(x1(:,2), x2(:,2)).*pairwiseDiff(x1(:,2), x2(:,2))).*K ];
        K_tilde = [ K, dKx2; dKx1, ddK ];
    elseif strcmp(Kerneltype, 'Matern')
        sigma = 1;
        K = sigma*(1 + sqrt(5)/R*pdist2(x1, x2) + 5/(3*R^2)*pdist2(x1, x2).^2).*...
            exp(-sqrt(5)/R*pdist2(x1, x2));
        dKx1 = [ exp(-sqrt(5)/R*pdist2(x1, x2)).*(5/R^2*pairwiseDiff(x1(:,1), x2(:,1)).*...
                    (2/3*pdist2(x1, x2) - 1 - sqrt(5)/(3*R)*pdist2(x1, x2)));
                 exp(-sqrt(5)/R*pdist2(x1, x2)).*(5/R^2*pairwiseDiff(x1(:,2), x2(:,2)).*...
                    (2/3*pdist2(x1, x2) - 1 - sqrt(5)/(3*R)*pdist2(x1, x2))) ];
        dKx2 = [ -(exp(-sqrt(5)/R*pdist2(x1, x2)).*(5/R^2*pairwiseDiff(x1(:,1), x2(:,1)).*...
                    (2/3*pdist2(x1, x2) - 1 - sqrt(5)/(3*R)*pdist2(x1, x2)))), ...
                 -(exp(-sqrt(5)/R*pdist2(x1, x2)).*(5/R^2*pairwiseDiff(x1(:,1), x2(:,1)).*...
                    (2/3*pdist2(x1, x2) - 1 - sqrt(5)/(3*R)*pdist2(x1, x2)))) ];
        ddK = [ exp(-sqrt(5)/R*pdist2(x1, x2)).*(...
                    (-10/(3*R^2) + 5*sqrt(5)/(3*R^3))*pdist2(x1,x2) + (10/(3*R^2) - 5*sqrt(5)/(3*R^3))* ...
                    pairwiseDiff(x1(:,1),x2(:,1)).^2./(pdist2(x1,x2) + 1e-6) + 5/R^2) + ...
                    sqrt(5)/R*pairwiseDiff(x1(:,1),x2(:,1))./(pdist2(x1,x2) + 1e-6).*dKx1(1:N,:), ...
                exp(-sqrt(5)/R*pdist2(x1, x2)).*(...
                    (-10/(3*R^2) + 5*sqrt(5)/(3*R^3))*pairwiseDiff(x1(:,1),x2(:,1)).*pairwiseDiff(x1(:,2),x2(:,2)))./(pdist2(x1,x2) + 1e-6) + ...
                    sqrt(5)/R*pairwiseDiff(x1(:,2),x2(:,2))./(pdist2(x1,x2) + 1e-6).*dKx1(1:N,:);
                exp(-sqrt(5)/R*pdist2(x1, x2)).*(...
                    (-10/(3*R^2) + 5*sqrt(5)/(3*R^3))*pairwiseDiff(x1(:,1),x2(:,1)).*pairwiseDiff(x1(:,2),x2(:,2)))./(pdist2(x1,x2) + 1e-6) + ...
                    sqrt(5)/R*pairwiseDiff(x1(:,1),x2(:,1))./(pdist2(x1,x2) + 1e-6).*dKx1(N+1:end,:), ...
                exp(-sqrt(5)/R*pdist2(x1, x2)).*(...
                    (-10/(3*R^2) + 5*sqrt(5)/(3*R^3))*pdist2(x1,x2) + (10/(3*R^2) - 5*sqrt(5)/(3*R^3))* ...
                    pairwiseDiff(x1(:,2),x2(:,2)).^2./(pdist2(x1,x2) + 1e-6) + 5/R^2) + ...
                    sqrt(5)/R*pairwiseDiff(x1(:,2),x2(:,2))./(pdist2(x1,x2) + 1e-6).*dKx1(N+1:end,:)];
        K_tilde = [ K, dKx2; dKx1, ddK ];
    else
        error('Kernel type not recognized')
    end
end

function out = pairwiseDiff(x1, x2)
    % x1, x2 column vectors
    out = zeros(size(x1,1), size(x2,1));
    for ii = 1:size(x1,1)
        for jj = 1:size(x2,1)
            out(ii,jj) = x1(ii) - x2(jj);
        end
    end
end