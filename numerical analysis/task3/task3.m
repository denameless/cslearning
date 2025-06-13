clear; clc; format long;

tol = 1e-8;        % 停止精度
max_iter = 100;    % 最大迭代次数

% --- 定义不同的初始猜测值 ---
initial_guesses = {
    [0.1; 0.1; -0.1],
    [0.5; 0.0; -0.5],
    [1.0; -0.2; 0.2]
};

% --- 循环求解并显示结果 ---
fprintf('牛顿法求解结果:\n');

for i = 1:length(initial_guesses)
    x0 = initial_guesses{i};
    fprintf('\n--- 初始值: [%.2f, %.2f, %.2f] ---\n', x0(1), x0(2), x0(3));
    
    % 调用牛顿法求解器
    [x_sol, ~, xx_hist] = newtons(@problem_equations, x0, tol, max_iter);
    
    num_iterations = size(xx_hist, 1) - 1;
    last_step_norm = norm(xx_hist(end,:) - xx_hist(end-1,:));

    fprintf('迭代次数: %d\n', num_iterations);
    fprintf('解 x = [%.12f, %.12f, %.12f]\n', x_sol(1), x_sol(2), x_sol(3));
    fprintf('最后一步误差 ||x(k)-x(k-1)|| = %e\n', last_step_norm);
end

% newtons函数：实现牛顿法迭代
function [x,fx,xx] = newtons(f,x0,TolX,MaxIter,varargin)
    h = 1e-4;
    fx = feval(f,x0,varargin{:});
    
    if nargin < 4, MaxIter = 100; end
    if nargin < 3, TolX = 1e-8; end
    
    xx(1,:) = x0(:).';

    for k = 1:MaxIter
        J = jacob(f,xx(k,:),h,varargin{:}); 
        dx = -J\fx(:); 
        
        xx(k + 1,:) = xx(k,:) + dx.';
        fx = feval(f,xx(k + 1,:),varargin{:});

        if norm(dx) < TolX, break; end
    end
    x = xx(k + 1,:);
end

% jacob函数：用中心差分法计算雅可比矩阵
% (源自 "APPLIED NUMERICAL METHODS USING MATLAB", P192)
function g = jacob(f,x,h,varargin)
    h2 = 2*h;
    N = length(x);
    I = eye(N);
    for n = 1:N
        g(:,n) = (feval(f,x + I(n,:)*h,varargin{:}) ...
                 -feval(f,x - I(n,:)*h,varargin{:})) / h2;
    end
end

function F = problem_equations(x)
    F = zeros(3,1);
    F(1) = 3*x(1) - cos(x(2)*x(3)) - 0.5;
    F(2) = x(1)^2 - 81*(x(2) + 0.1)^2 + sin(x(3)) + 1.06;
    F(3) = exp(-x(1)*x(2)) + 20*x(3) + 10/3*pi - 1;
end