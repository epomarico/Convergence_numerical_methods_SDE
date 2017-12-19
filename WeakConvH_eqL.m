% WEAK CONVERGENCE OF THE HEUN NUMERICAL METHOD
% FOR A LINEAR SDE WITH AN ADDITIVE NOISE 
%
% This program tests the weak convergence of the Heun method for the numerical 
% solution of the following stochastic differential equation
%
% dX = f(X) dt + sigma*dW, X(0) = Xzero,
% with f(X)=alpha X, sigma=const and Xzero = 1.
%
% To study the weak convergence of the methods we calculate 
% the error |E(phi(X_T)) - E(phi(X(T)))| which is the absolute value of the differences 
% between the sample average of the value of a function test phi on the numerical solutions and on the exact solution
% at time T X_T and X(T). We will consider as function test phi=x and
% phi=x^2

clear all
  
% Initialize random number generator
randn('state',100) 
  
%  Equation parameters
alpha = 1.0; sigma =0.001; Xzero = 1.0; 
% Time parameters
T = 1.0; dtvals = 2.^([1:5]-10);
% Number of brownian trajectories
m = 1000000; 

% Choose the probability distribution of the brownian steps
% The variable method indicates the kind of distribution for the brownian steps
% (method=0: steps with gaussian distribution with centre 0 and
% variance the value of the step;
% method =1: discrete steps =+-1 with probbility +-0.5 respectively)
method=0; 
  
% Initialize vectors for the errors 
XH = zeros(5,1); X2H = zeros(5,1);

for p = 1 : 5           % consider 5 different time-steps 
  dt = dtvals(p);  l = T/dt;   % set the value of the time-step Dt and the number of steps to consider to arrive to T    
  XtH = Xzero*ones(m,1);       % initialize the mx1 vectors X(T) for the three algorithms  
                        
  for j = 1 : l
    if (method == 0) dW = sqrt(dt)*randn(m,1); %m x 1 vector of Brownian increments according to the value of 'method'
      else dW = sqrt(dt)*sign(randn(m,1));
    end
    K = zeros(m,1);
    K = XtH + alpha*XtH*dt + sqrt(dt)*sigma*dW; %First step in Heun method
    XtH = XtH + 0.5*dt*alpha*(XtH + alpha*K) + sqrt(dt)*sigma*dW; %Second step in Heun method
  end
  
  XH(p) = mean (XtH);  X2H(p) = mean (XtH.^2); % Error of EM method at t=T for phi=x and phi=x^2
 
end

%Evaluation of errors for phi=x and phi=x^2
%Exact solutions of the equation
EXth=Xzero*exp(alpha*T);
EX2th=(exp(2*alpha*T)*(Xzero+((sigma^2)/(2*alpha))))-(exp(alpha*T)*((sigma^2)/(2*alpha)));
%Errors for Heun algorithm
XerrH = abs(XH - EXth); 
Xerr2H = abs(X2H - EX2th); 


%Fitting of the errors for the three algorithms
H1=polyfit(log10(dtvals),log10(XerrH)',1);
slope=H1(1)
H2=polyfit(log10(dtvals),log10(Xerr2H)',1);
slope=H2(1)

% Plotting the graph of weak convergence for the three methods for phi=x
% and phix^2
close all
str = {'\fontsize{14}dX = f(X)dt +\sigma dW','X(0)=1, f(X)=\alpha X,\alpha =1,','\sigma=0.001, M=10^6'};

%Heun phi=x
figure;
subplot(1,2,1);
loglog(dtvals,XerrH','b*-'); hold on;
loglog(dtvals,dtvals.^2,'r--'); hold off; %  Include a reference slope of 2   
%axis([1e-3 1e-1 1e-4 1])
xlabel('\fontsize{14}\Delta t'); ylabel('\fontsize{14}|E(X_N) - E(X(T))|');
title('\fontsize{14}Weak convergence (\phi(x)=x)');
dim = [0.22 0.03 0.26 0.25];
annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','none');
legend('\fontsize{14}Heun','\fontsize{14}ref. slope 2');
legend('Location','Northwest'); legend('boxoff');

%Heun phi=x^2
subplot(1,2,2);
loglog(dtvals,Xerr2H','b*-');hold on;
loglog(dtvals,dtvals.^2,'r--');hold off; %  Include a reference slope of 2   
%axis([1e-3 1e-1 1e-4 1])
xlabel('\fontsize{14}\Delta t'); ylabel('\fontsize{14}|E(X^2_N) - E(X^2(T))|');
title('\fontsize{14}Weak convergence (\phi(x)=x^2)');
dim2 = [0.67 0.03 0.26 0.25];
annotation('textbox',dim2,'String',str,'FitBoxToText','on','EdgeColor','none');
legend('\fontsize{14}Heun','\fontsize{14}ref. slope 2');
legend('Location','Northwest'); legend('boxoff'); 




