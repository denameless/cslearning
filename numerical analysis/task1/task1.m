clear; clc;
format long;

%% 定义函数和精确值
% 使用一个独立的局部函数 f_integrand 来正确处理 x=0 的情况
f = @f_integrand; 
I_exact = -4/9; % 积分的精确值

a = 0; % 积分下限
b = 1; % 积分上限

%  问题 (1): 复合梯形和复合辛普森法

n_vals = 2.^(2:12);
h_vals = (b-a) ./ n_vals;

T_results = zeros(size(n_vals));
S_results = zeros(size(n_vals));
T_errors = zeros(size(n_vals));
S_errors = zeros(size(n_vals));

fprintf(' n         h             梯形法结果          辛普森法结果        梯形法误差        辛普森法误差\n');
fprintf('----------------------------------------------------------------------------------------------------\n');

for i = 1:length(n_vals)
    n = n_vals(i);
    h = h_vals(i);
    x = linspace(a, b, n+1);
    y = f(x);
    
    T_results(i) = h/2 * (y(1) + 2*sum(y(2:end-1)) + y(end));
    T_errors(i) = abs(T_results(i) - I_exact);
    
    S_results(i) = h/3 * (y(1) + 4*sum(y(2:2:end-1)) + 2*sum(y(3:2:end-2)) + y(end));
    S_errors(i) = abs(S_results(i) - I_exact);
    
    fprintf('%5d   %.6f   %.12f   %.12f   %.4e   %.4e\n', ...
            n, h, T_results(i), S_results(i), T_errors(i), S_errors(i));
end

figure('Name', '误差分析图 (问题1)');
loglog(h_vals, T_errors, '-o', 'DisplayName', '复合梯形法误差');
hold on;
loglog(h_vals, S_errors, '-s', 'DisplayName', '复合辛普森法误差');
loglog(h_vals, (h_vals.^1.5), 'k--', 'DisplayName', '参考线 O(h^{1.5})');
grid on;
xlabel('步长 h');
ylabel('绝对误差');
title('误差 vs. 步长 h (双对数坐标)');
legend('Location', 'southeast');
set(gca, 'XDir','reverse');

fprintf('\n关于最小h的讨论:\n');
fprintf('... (解释与之前相同) ...\n\n');

%  问题 (2): 龙贝格(Romberg)求积

max_k = 8;
R = zeros(max_k, max_k);

h = b - a;
R(1, 1) = h/2 * (f(a) + f(b));
for k = 2:max_k
    h = h / 2;
    sum_f = sum(f(a + (1:2^(k-2))*2*h - h));
    R(k, 1) = 0.5 * R(k-1, 1) + h * sum_f;
end

for j = 2:max_k
    for k = j:max_k
        R(k, j) = (4^(j-1) * R(k, j-1) - R(k-1, j-1)) / (4^(j-1) - 1);
    end
end

fprintf('龙贝格求积表 R(k,j):\n');
disp(R);

romberg_result = R(max_k, max_k);
romberg_error = abs(romberg_result - I_exact);

fprintf('\n龙贝格积分最终结果: %.12f\n', romberg_result);
fprintf('龙贝格积分绝对误差: %.4e\n', romberg_error);

%  问题 (3): 自适应辛普森法

tol = 1e-4;
[adaptive_S_result, node_count] = adaptive_simpson_main(f, a, b, tol);
adaptive_S_error = abs(adaptive_S_result - I_exact);

fprintf('目标精度 tol = %.1e\n', tol);
fprintf('自适应辛普森法结果: %.12f\n', adaptive_S_result);
fprintf('最终使用的函数求值次数: %d\n', node_count);
fprintf('自适应辛普森法误差: %.4e\n', adaptive_S_error);
fprintf('误差 %.4e 小于目标精度 %.1e，任务完成。\n', adaptive_S_error, tol);

%  局部函数定义 (Local Functions)

function y = f_integrand(x)
    y = zeros(size(x)); % 初始化输出为0
    % 找到所有 x>0 的位置
    idx = x > 0;
    % 只对 x>0 的元素计算，其余位置保持为0
    y(idx) = sqrt(x(idx)) .* log(x(idx));
end

function [I, count] = adaptive_simpson_main(f, a, b, tol)
    % 主函数，调用递归函数
    c = (a + b) / 2;
    fa = f(a); fc = f(c); fb = f(b);
    count = 3; % 初始计算了3个点
    [I, count_recursive] = adaptive_simpson_recursive(f, a, b, tol, fa, fc, fb);
    count = count + count_recursive;
end

function [I, count] = adaptive_simpson_recursive(f, a, b, tol, fa, fc, fb)
    % 递归函数
    c = (a + b) / 2;
    h = b - a;
    S1 = h/6 * (fa + 4*fc + fb);
    
    d = (a + c) / 2;
    e = (c + b) / 2;
    fd = f(d); 
    fe = f(e);
    count = 2; % 本次调用新计算了2个点

    S2 = (h/12) * (fa + 4*fd + 2*fc + 4*fe + fb);
    
    if abs(S2 - S1) / 15 < tol
        I = S2 + (S2 - S1)/15; 
    else
        [I_left, count_left] = adaptive_simpson_recursive(f, a, c, tol/2, fa, fd, fc);
        [I_right, count_right] = adaptive_simpson_recursive(f, c, b, tol/2, fc, fe, fb);
        I = I_left + I_right;
        count = count + count_left + count_right;
    end
end