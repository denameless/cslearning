clear; clc;
format long;

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

    T_results(i) = trpzds(f, a, b, n);
    S_results(i) = Smpsns(f, a, b, n);

    T_errors(i) = abs(T_results(i) - I_exact);
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

%  问题 (2): 龙贝格(Romberg)求积

tol_romberg = 1e-12; % 设置精度目标
max_k = 8;           % 最大迭代次数，对应 Rmbrg 的参数 K

[romberg_result, R, ~, ~] = Rmbrg(f, a, b, tol_romberg, max_k);

romberg_error = abs(romberg_result - I_exact);

fprintf('\n龙贝格求积表 R(k,j):\n');
disp(R);

fprintf('\n龙贝格积分最终结果: %.12f\n', romberg_result);
fprintf('龙贝格积分绝对误差: %.4e\n', romberg_error);

%  局部函数定义

function y = f_integrand(x)
    y = zeros(size(x)); % 初始化输出为0
    % 找到所有 x>0 的位置
    idx = x > 0;
    % 只对 x>0 的元素计算，其余位置保持为0
    y(idx) = sqrt(x(idx)) .* log(x(idx));
end

function INTf=Smpsns(f,a,b,N,varargin)
if nargin<4, N=100; end
if abs(b-a)<1e-12|N<=0, INTf=0; return; end
if mod(N,2)~=0, N=N+1; end 
h=(b-a)/N; x=a+[0:N]*h; 
fx=feval(f,x,varargin{:}); 
fx(find(fx==inf))=realmax; fx(find(fx==-inf))=-realmax;
kodd=2:2:N; keven=3:2:N-1; 
INTf=h/3*(fx(1)+fx(N+1)+4*sum(fx(kodd))+2*sum(fx(keven)));
end

function INTf=trpzds(f,a,b,N)
if abs(b-a)<eps|N<=0, INTf=0; return; end
h=(b-a)/N; x=a+[0:N]*h; fx=feval(f,x); 
INTf=h*((fx(1)+fx(N+1))/2+sum(fx(2:N))); 
end

function [x,R,err,N]=Rmbrg(f,a,b,tol,K)
%construct Romberg table to find definite integral of f over [a,b]
h=b-a; N=1;
if nargin<5, K=10; end
R(1,1)=h/2*(feval(f,a)+feval(f,b));
for k=2:K
h=h/2; N=N*2;
R(k,1)=R(k-1,1)/2 +h*sum(feval(f,a+[1:2:N-1]*h)); %Eq.(5.7.1)
tmp=1;
for n=2:k
tmp= tmp*4;
R(k,n)= (tmp*R(k,n-1)-R(k-1,n-1))/(tmp-1); %Eq.(5.7.3)
end
err= abs(R(k,k-1)-R(k-1,k-1))/(tmp-1); %Eq.(5.7.4)
if err<tol, break; end
end
x=R(k,k);
end