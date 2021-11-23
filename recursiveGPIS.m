%% Recursive GPIS

% [x1, x2, button] = ginput(30);
% % mouse left: space; 
% %       middle: countour; 
% %       right: obstacle.
% x = 4*[x1'; x2'];
% y = button' - 2;

f_user = figure;

figure(f_user)
[x1, x2, button] = ginput(10);
% mouse left: space; 
%       middle: countour; 
%       right: obstacle.
x_first = 4*[x1'; x2']';
y_first = button - 2;

R = 1;
[ X1, X2 ] = meshgrid(linspace(0, 4, 50), linspace(0, 4, 50));
basVect = [ X1(:), X2(:) ];

Kss = 2*abs(pdist2(basVect, basVect).^3) - 3*R*pdist2(basVect, basVect).^2 + R^3;
Ks = 2*abs(pdist2(basVect, x_first).^3) - 3*R*pdist2(basVect, x_first).^2 + R^3;
K = 2*abs(pdist2(x_first, x_first).^3) - 3*R*pdist2(x_first, x_first).^2 + R^3;

ypred = ( (Ks / (K + 0*eye(size(K,1)))) * (y_first))';
ysd = (Kss - Ks/(K + 0*eye(size(K,1)))*Ks');

f1 = figure;
figure(f1)
cla
surface(X1, X2, reshape(ypred, 50, 50) - max(ypred), 'FaceColor','interp','EdgeColor','interp');
hold on
plot(x_first(y_first==1,1), x_first(y_first==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x_first(y_first==0,1), x_first(y_first==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x_first(y_first==-1,1), x_first(y_first==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
contour(X1, X2, reshape(ypred, 50, 50), [0,0]);
colormap('gray');
title('Environment modeling')
xlabel('x_1 [m]')
ylabel('x_2 [m]')
colorbar('Ticks', [min(ypred)-max(ypred), 0], 'TickLabels', {'Free', 'Obstacle'})
hold off

% Saving all the points for plotting them later
x_all = x_first;
y_all = y_first;

% Initialize RGP

K = kernelFnct(basVect, basVect, R);
% mu_g_old = -1 + zeros(size(basVect, 1),1); % Initial condition on ypred
mu_g_old = ypred';
% Cg_old = 10e2*eye(size(basVect,1));
Cg_old = ysd;
% K_inv = inv(K);


grid on;
for ii = 1:5
    % At each iteration ask the user a new batch
    figure(f_user);
    [x1, x2, button] = ginput(10);
    % mouse left: space; 
    %       middle: countour; 
    %       right: obstacle.
    x_batch = 4*[x1'; x2']';
    y_batch = button - 2;
    x_all = [ x_all; x_batch ];
    y_all = [ y_all; y_batch ];
    close(f_user);
    
    % Inference
    Ks = kernelFnct(x_batch, basVect, R);
    Kss = kernelFnct(x_batch, x_batch, R);
    J = Ks/K;
    mu_p = J*(mu_g_old);
    B = Kss - J*Ks';
    Cp = B + J*Cg_old*J';
    % Update
    G = Cg_old*J'/(Cp + 1e-4*eye(size(Cp))); % gain matrix, inversion of a matrix big as the batch used
    mu_g = mu_g_old + G*(y_batch - mu_p);
    Cg = Cg_old - G*J*Cg_old;
    mu_g_old = mu_g;
    Cg_old = Cg;
    
    figure(f1)
    cla
    surface(X1, X2, reshape(mu_g, 50, 50) - max(mu_g), 'FaceColor','interp','EdgeColor','interp');
    hold on
    plot(x_all(y_all==1,1), x_all(y_all==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
    plot(x_all(y_all==0,1), x_all(y_all==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
    plot(x_all(y_all==-1,1), x_all(y_all==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
    contour(X1, X2, reshape(mu_g, 50, 50), [0,0]);
    colormap('gray');
    title('Environment modeling')
    xlabel('x_1 [m]')
    ylabel('x_2 [m]')
    colorbar('Ticks', [min(mu_g)-max(mu_g), 0], 'TickLabels', {'Free', 'Obstacle'})
    hold off
    
    f_user = figure;
end
    
    
    
function K = kernelFnct(x1, x2, R)

    K = 2*abs(pdist2(x1, x2).^3) - 3*R*pdist2(x1, x2).^2 + R^3; % Basis function
    
end
        