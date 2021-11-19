

% hermiteH([ 0 1 2 3 ], [1.5; 2; 3; 4; 10])
% 2^(4/2 - 1)*hermiteH(4, sqrt(2)*10)
hermiteH([0, 1, 2, 3, 4, 5], [10; 2; 4])

function [y, H0] = hermiteH(n, x)

    % Hermite polynomial function.
    % n is the row vector of the desired degrees of the hermite polynomial,
    % x is the column vector of the input. 
    x = sqrt(2)*x;
    y = zeros(size(x,1), length(n));
    for j = 1:length(n)
        H0 = 1;
        H1 = [1,0];
        for i = 1:n(j)
            H0_1 = i * H0;
            H0_2 = conv ( [0,0,1], H0_1 );
            H2 = conv( [1,0], H1 ) - H0_2;
            H0 = H1;
            H1 = H2;
        end
        for i = 1:n(j)+1
            y(:,j) = y(:,j) + x.^(n(j)+1-i) * H0(i);
        end
        y(:,j) = 2^(n(j)/2)*y(:,j);
    end
end