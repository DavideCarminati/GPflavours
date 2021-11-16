%% 2D case
% In 2D, I need to specify a Y vector of (dim+1)N points, since I have to
% specify a 2D gradient this time! Also, the kernel derivatives are vectors
% if referred to a single point.
clear
noise = 1e-3;
lambda = 30;
numTest = 50;
wallPoints = 10;
R = 5;
circleRadius = 0.5;
x = [2.5, 2.5] + circleRadius * [cos(-pi:pi/4:(pi-1e-3))', sin(-pi:pi/4:(pi-1e-3))'];
x2 = [4, 1] + 1.25*circleRadius * [cos(-pi:pi/4:(pi-1e-3))', sin(-pi:pi/4:(pi-1e-3))'];

% x = [ x; x2 ];

% Square room with circle obstacle:
% x = [ x;
%       linspace(0.6, 4.4, wallPoints)', 0.5*ones(wallPoints,1);
%       linspace(0.6, 4.4, wallPoints)', 4.5*ones(wallPoints,1);
%       0.5*ones(wallPoints,1), linspace(0.6, 4.4, wallPoints)';
%       4.5*ones(wallPoints,1), linspace(0.6, 4.4, wallPoints)' ];

% x = [ x;
%       0, 2;
%       2, 0;
%       5, 2;
%       2, 5 ];

% x = [ 2, 4;
%       3, 3;
%       4.5, 3.5 ];

[ X1, X2 ] = meshgrid(linspace(0,5,numTest), linspace(0,5,numTest));
X = [ X1(:), X2(:) ];

[ K_tilde, K ] = kernelFnct2D(x, x, R);
[ Ks_tilde, Ks ] = kernelFnct2D(X, x, R);

y = zeros(size(x, 1), 1);
% y = [ 0; 0; 0; 0; 0; 0; 0; 0 ];

dy = [ 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2; -1; -sqrt(2)/2; % Gradient along x1
        -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2 ]; % Gradient along x2

% dy = [ 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2; -1; -sqrt(2)/2; % Gradient along x1
%         1; sqrt(2)/2; 0; -sqrt(2)/2; -1; -sqrt(2)/2; 0; sqrt(2)/2 ]; % Gradient along x2

% dy = [ -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2;
%         0; -sqrt(2)/2; -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2 ];

% dy = [ dy(1:length(dy)/2); dy(1:length(dy)/2); dy(length(dy)/2+1:end); dy(length(dy)/2+1:end) ];

% Square room with circle obstacle:
% dy = [ 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2; -1; -sqrt(2)/2; -ones(wallPoints,1); ones(wallPoints,1); zeros(2*wallPoints,1);
%         -1; -sqrt(2)/2; 0; sqrt(2)/2; 1; sqrt(2)/2; 0; -sqrt(2)/2; zeros(2*wallPoints,1); ones(wallPoints,1); -ones(wallPoints,1)]; 

y = exp(-y*lambda);% + noise*randn(size(x, 1), 1);
mu_g = Ks_tilde / K_tilde * [y; dy];

% Without gradient information
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
grad = -mu_g(numTest^2+1:end)./normlz(doublingIndex);

% Rotate the gradients so that they are normal to surfaces
theta = -pi/2;
rot = [ cos(theta)*eye(numTest^2), sin(theta)*eye(numTest^2); ...
        -sin(theta)*eye(numTest^2), cos(theta)*eye(numTest^2) ];
grad = rot*grad;

theta = pi/2;
rot = [ cos(theta)*eye(length(dy)/2), sin(theta)*eye(length(dy)/2); ...
        -sin(theta)*eye(length(dy)/2), cos(theta)*eye(length(dy)/2) ];
dy_normal = rot*dy;

figure
hsurf = surface(X1, X2, reshape(dist, numTest, numTest) - max(dist), 'FaceColor','interp','EdgeColor','interp');
% colormap gray;
% imagesc(linspace(0,5,numTest), linspace(0,5,numTest), reshape(dist, numTest, numTest));
hold on
plot(x(:,1), x(:,2), '.','markersize',28,'color',[.7 0.3 0]); %Interior points
quiver(x(:,1), x(:,2), dy_normal(1:length(y)), dy_normal(length(y)+1:end), 'g');
% quiver(X(:,1), X(:,2), grad(1:numTest^2), grad(numTest^2+1:end), 'w');
% set(gca, 'Layer', 'top')
hsurf.Annotation.LegendInformation.IconDisplayStyle = 'off';
legend('Obstacle border', 'Normal to border')
xlabel('x_1 [m]')
ylabel('x_2 [m]')
xlim([0, 5])
ylim([0, 5])
% view(3)

% figure
% surface(X1, X2, reshape(dist2, numTest, numTest), 'FaceColor','interp','EdgeColor','interp');
% hold on
% plot(x(:,1), x(:,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
% xlabel('x_1 [m]')
% ylabel('x_2 [m]')
% title('Without grad info');

function [ K_tilde, K ] = kernelFnct2D(x1, x2, R)

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
         
    ddK = [ -6*pairwiseDiff(x1(:,1), x2(:,1)).*pairwiseDiff(x1(:,1), x2(:,1))./(pdist2(x1, x2)+1e-4) + pdist2(x1, x2) - R, ...
            -6*pairwiseDiff(x1(:,1), x2(:,1)).*pairwiseDiff(x1(:,2), x2(:,2))./(pdist2(x1, x2)+1e-4);
            -6*pairwiseDiff(x1(:,2), x2(:,2)).*pairwiseDiff(x1(:,1), x2(:,1))./(pdist2(x1, x2)+1e-4), ...
            -6*pairwiseDiff(x1(:,2), x2(:,2)).*pairwiseDiff(x1(:,2), x2(:,2))./(pdist2(x1, x2)+1e-4) + pdist2(x1, x2) - R ];
    K_tilde = [ K, dKx2; dKx1, ddK ];
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