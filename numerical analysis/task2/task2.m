clear; clc;
format longG;

A = [10, -7,  0,  1;
     -3,  2.099999, 6,  2;
      5, -1,  5, -1;
      2,  1,  0,  2];

b = [8;
     5.900001;
     5;
     1];

% 方法一: LU 分解法

fprintf('--- 方法一: LU 分解法 ---\n\n');

[L, U, P] = lu_dcmp(A);

y = L \ (P * b);
x_lu = U \ y;

% 计算行列式: det(A) = det(P')*det(L)*det(U)

detA_lu = det(P) * prod(diag(U));

disp('LU 分解得到的矩阵 L:');
disp(L);
disp('LU 分解得到的矩阵 U:');
disp(U);
disp('LU 分解得到的解向量 x:');
disp(x_lu);
fprintf('通过 LU 分解计算的 det(A) = %.12f\n', detA_lu);

% 方法二: 列主元高斯消去法

fprintf('--- 方法二: 列主元高斯消去法 ---\n\n');

% 调用修改后的 gauss 函数，获取解和行交换历史
[x_gauss, pivot_history] = gauss_with_pivot_tracking(A, b);

disp('列主元法中的行交换次序 (最终状态):');
disp('原始行 -> 最终行');
original_rows = (1:size(A,1))';
final_rows = pivot_history';
disp(table(original_rows, final_rows));

disp('列主元法得到的解向量 x:');
disp(x_gauss);
detA_gauss = det(A);
fprintf('通过高斯消去法计算的 det(A) = %.12f\n', detA_gauss);

% 结果比较

fprintf('--- 结果比较 ---\n\n');

fprintf('两种方法得到的解向量 x 的差的范数: %e\n', norm(x_lu - x_gauss));
fprintf('两种方法得到的 det(A) 的差: %e\n\n', abs(detA_lu - detA_gauss));

% 局部函数

function [L,U,P] = lu_dcmp(A)
    NA = size(A, 1);
    L = eye(NA);
    U = zeros(NA);
    for k = 1:NA
        for j = k:NA
            U(k,j) = A(k,j) - L(k, 1:k-1) * U(1:k-1, j);
        end
        if abs(U(k,k)) < eps
            error('主元为零，分解失败');
        end
        for i = k+1:NA
            L(i,k) = (A(i,k) - L(i, 1:k-1) * U(1:k-1, k)) / U(k,k);
        end
    end
    P = eye(NA);
end

function [x, p_order] = gauss_with_pivot_tracking(A,B)
    % The sizes of matrices A,B are supposed to be NA x NA and NA x NB.
    % This function solves Ax = B by Gauss elimination algorithm.
    % MODIFIED to return the pivot order.
    NA = size(A,2); [NB1,NB] = size(B);
    if NB1 ~= NA, error('A and B must have compatible dimensions'); end
    N = NA + NB; AB = [A(1:NA,1:NA) B(1:NA,1:NB)]; % Augmented matrix
    epss = eps*ones(NA,1);
    
    p_order = 1:NA; % 添加初始化行顺序追踪器
    
    for k = 1:NA
        %Scaled Partial Pivoting at AB(k,k) by Eq.(2.2.20)
        [~,kx] = max(abs(AB(k:NA,k)));
        if abs(AB(k+kx-1, k)) < eps, error('Singular matrix and No unique solution'); end
        mx = k + kx - 1;
        if kx > 1 % Row change if necessary
            tmp_row = AB(k,k:N);
            AB(k,k:N) = AB(mx,k:N);
            AB(mx,k:N) = tmp_row;
            % 追踪行交换
            p_order([k, mx]) = p_order([mx, k]);
        end
        % Gauss forward elimination
        AB(k,k + 1:N) = AB(k,k+1:N)/AB(k,k);
        AB(k,k) = 1; %make each diagonal element one
        for m = k + 1: NA
            AB(m,k+1:N) = AB(m,k+1:N) - AB(m,k)*AB(k,k+1:N); %Eq.(2.2.5)
            AB(m,k) = 0;
        end
    end
    %backward substitution for a upper-triangular matrix eqation
    % having all the diagonal elements equal to one
    x = zeros(NA, NB); % 初始化解向量
    x(NA,:) = AB(NA,NA+1:N);
    for m = NA-1: -1:1
        x(m,:) = AB(m,NA + 1:N)-AB(m,m + 1:NA)*x(m + 1:NA,:); %Eq.(2.2.7)
    end
end