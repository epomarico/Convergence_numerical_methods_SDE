% WEAK CONVERGENCE OF THE EULER-MARUYAMA AND MILSTEIN NUMERICAL METHODS
% FOR A NONLINEAR SDE
%
% This program tests the strong convergence of the Euler-Maruyama (EM) method,
% the Milstein (M) one, as well as its derivative-free (M2), for the numerical 
% solution of the following stochastic differential equation
%
% dX=(0.5*X(t)+sqrt(X(t)^2 +1)) dt + sqrt(X(t)^2 +1)*dW(t), X(0)=X0,
% with Xzero = 1.
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
Xzero = 1.0; 
% Time parameters
T = 1.0; dtvals = 2.^([1:5]-10);
% Number of brownian trajectories
m = 1000000; 
% Function g(X) which multiples dW in the equation
g=@(x) sqrt(x.^2 + 1);
dx=0.001; %spatial step for derivative calculation

% Choose the probability distribution of the brownian steps
% The variable method indicates the kind of distribution for the brownian steps
% (method=0: steps with gaussian distribution with centre 0 and
% variance the value of the step;
% method =1: discrete steps =+-1 with probbility +-0.5 respectively)
method=1; 
  
% Initialize vectors for the errors for the three algorithms
XEM = zeros(5,1); X2EM = zeros(5,1);               
XM = zeros(5,1); X2M = zeros(5,1); 
XM2 = zeros(5,1); X2M2 = zeros(5,1); 

for p = 1 : 5                   % consider 5 different time-steps 
  dt = dtvals(p); l = T/dt;     % set the value of the time-step Dt and the number of steps to consider to arrive to T
  XtEM = Xzero*ones(m,1);  XtM = Xzero*ones(m,1); XtM2 = Xzero*ones(m,1);    % initialize the mx1 vectors X(T) for the three algorithms   
          
  for j = 1 : l
    if (method == 0) dW = sqrt(dt)*randn(m,1); %m x 1 vector of Brownian increments according to the value of 'method'
      else dW = sqrt(dt)*sign(randn(m,1));
    end
    XtEM = XtEM + (0.5*XtEM + sqrt(XtEM.^2 +1))*dt + sqrt(XtEM.^2 +1).*dW; %Euler - Maruyama method
    
    gtM=sqrt(XtM.^2 +1);gtM_der=XtM./sqrt(XtM.^2 +1); % Analythical expression of the derivative for Milstein method
    XtM = XtM + (0.5*XtM + sqrt(XtM.^2 +1))*dt + sqrt(XtM.^2 +1).*dW +0.5*gtM.*gtM_der.*(dW.^2-dt); %Milstein method
    
    gp=(1/(2*dx))*(g(XtM2+dx)-g(XtM2-dx));       %Calculation of the derivative of g(X)
    XtM2 = XtM2 + (0.5*XtM2 + sqrt(XtM2.^2 +1))*dt  +g(XtM2).*dW + 0.5*gp.*g(XtM2).*(dW.^2-dt); % Derivative-free Milstein method
    
  end
  
  XEM(p) = mean (XtEM); X2EM(p) = mean (XtEM.^2);  % Error of EM method at t=T for phi=x and phi=x^2
  XM(p) = mean (XtM); X2M(p) = mean (XtM.^2);      % Error of M method at t=T for phi=x and phi=x^2
  XM2(p) = mean (XtM2); X2M2(p) = mean (XtM2.^2);  % Error of M2 method at t=T for phi=x and phi=x^2
end

%Evaluation of errors for phi=x and phi=x^2 

%Exact solutions of the equation
EXth=exp(T/2)*sinh(T+asinh(Xzero));
EX2th=-0.5+0.25*(exp(4*T+2*asinh(Xzero))+exp(-2*asinh(Xzero)));
%Errors for the EM algorithm
XerrEM = abs(XEM - EXth); X2errEM = abs(X2EM - EX2th);
%Errors for M algorithm
XerrM = abs(XM - EXth); X2errM = abs(X2M - EX2th); 
%Errors for M2 algorithm
XerrM2 = abs(XM2 - EXth); X2errM2 = abs(X2M2 - EX2th); 

%Fitting of the errors for the three algorithms
EM=polyfit(log10(dtvals),log10(XerrEM)',1);
slope_EM1=EM(1)
EM2=polyfit(log10(dtvals),log10(X2errEM)',1);
slope_EM2=EM2(1)

Mil=polyfit(log10(dtvals),log10(XerrM)',1);
slope_Mil1=Mil(1)
Mil2=polyfit(log10(dtvals),log10(X2errM)',1);
slope_Mil2=Mil2(1)

Mil_DF2=polyfit(log10(dtvals),log10(XerrM2)',1);
slope_MilDF21=Mil_DF2(1)
Mil_DF22=polyfit(log10(dtvals),log10(X2errM2)',1);
slope_MilDF22=Mil_DF22(1)

% Plotting the graph of weak convergence for the three methods for phi=x
% and phix^2
close all 
str = {'\fontsize{12}dX = (X/2+sqrt(X^2+1))dt', '+ sqrt(X^2+1) dW, X(0)=1, M=10^6'};

%EM phi=x
figure;
subplot(3,2,1);
loglog(dtvals,XerrEM','b*-'); hold on;
loglog(dtvals,dtvals,'r--'); hold off; %  Include a reference slope of 1.   
xlabel('\fontsize{14}\Delta t'); ylabel('\fontsize{14}|E(X_N) - E(X(T))|');
title('\fontsize{12}Weak convergence (\phi(x)=x)');
dim = [0.23 0.52 0.26 0.25];
annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','none');
dim1 = [0.05 0.71 0.26 0.25];
annotation('textbox',dim1,'String','\fontsize{14}a)','FitBoxToText','on','EdgeColor','none');
legend('\fontsize{14}Euler-Maruyama','\fontsize{14}ref. slope 1','Location','Northwest')
legend('boxoff')  

%EM phi=x^2
subplot(3,2,2);
loglog(dtvals,X2errEM','b*-');hold on;
loglog(dtvals,dtvals,'r--');hold off; %  Include a reference slope of 1.   
xlabel('\fontsize{14}\Delta t'); ylabel('\fontsize{14}|E(X^2_N) - E(X^2(T))|');
title('\fontsize{12}Weak convergence (\phi(x)=x)^2');
axis([1e-3 1e-1 1e-4 1e3]);
dim = [0.62 0.52 0.26 0.25];
annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','none','HorizontalAlignment','center');
dim1 = [0.5 0.71 0.26 0.25];
annotation('textbox',dim1,'String','\fontsize{14}b)','FitBoxToText','on','EdgeColor','none');

legend('\fontsize{14}Euler-Maruyama','\fontsize{14}ref. slope 1','Location','Northwest');
legend('boxoff'); 

%------------ 

%M phi=x
subplot(3,2,3);
loglog(dtvals,XerrM','b*-'); hold on; 
loglog(dtvals,dtvals,'r--'); hold off; %  Include a reference slope of 1.   
%axis([1e-3 1e-1 1e-4 1])
xlabel('\fontsize{14}\Delta t'); ylabel('\fontsize{14}|E(X_N) - E(X(T))|');
title('\fontsize{12}Weak convergence (\phi(x)=x)');
dim = [0.23 0.30 0.06 0.17];
annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','none');
dim1 = [0.05 0.41 0.26 0.25];
annotation('textbox',dim1,'String','\fontsize{14}c)','FitBoxToText','on','EdgeColor','none');
legend('\fontsize{14}Milstein','\fontsize{14}ref. slope 1','Location','Northwest');
legend('boxoff');

%M phi=x^2
subplot(3,2,4);
loglog(dtvals,X2errM','b*-'); hold on;
loglog(dtvals,dtvals,'r--'); hold off; %  Include a reference slope of 1.   
xlabel('\fontsize{14}\Delta t'); ylabel('\fontsize{14}|E(X^2_N) - E(X^2(T))|');
title('\fontsize{12}Weak convergence (\phi(x)=x^2)');
axis([1e-3 1e-1 1e-4 1e3]);
dim = [0.62 0.22 0.26 0.25];
annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','none','HorizontalAlignment','center');
dim1 = [0.5 0.41 0.26 0.25];
annotation('textbox',dim1,'String','\fontsize{14}d)','FitBoxToText','on','EdgeColor','none');
legend('\fontsize{14}Milstein','\fontsize{14}ref. slope 1','Location','Northwest');
legend('boxoff');  

% -----------

%M2 phi=x
subplot(3,2,5);
loglog(dtvals,XerrM2','b*-'); hold on; 
loglog(dtvals,dtvals,'r--'); hold off; %  Include a reference slope of 1.   
%axis([1e-3 1e-1 1e-4 1])
xlabel('\fontsize{14}\Delta t'); ylabel('\fontsize{14}|E(X_N) - E(X(T))|');
title('\fontsize{12}Weak convergence (\phi(x)=x)');
dim = [0.23 0.00 0.06 0.17];
annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','none');
dim1 = [0.05 0.11 0.26 0.25];
annotation('textbox',dim1,'String','\fontsize{14}e)','FitBoxToText','on','EdgeColor','none');
legend('\fontsize{14}Milstein (derivative-free)','\fontsize{14}ref. slope 1','Location','Northwest');
legend('boxoff');

%M2 phi=x^2
subplot(3,2,6);
loglog(dtvals,X2errM2','b*-'); hold on;
loglog(dtvals,dtvals,'r--'); hold off; %  Include a reference slope of 1.   
xlabel('\fontsize{14}\Delta t'); ylabel('\fontsize{14}|E(X^2_N) - E(X^2(T))|');
title('\fontsize{12}Weak convergence (\phi(x)=x^2)');
axis([1e-3 1e-1 1e-4 1e3]);
dim = [0.62 0.00 0.06 0.17];
annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','none','HorizontalAlignment','center');
dim1 = [0.5 0.11 0.26 0.25];
annotation('textbox',dim1,'String','\fontsize{14}f)','FitBoxToText','on','EdgeColor','none');
legend('\fontsize{14}Milstein (derivative-free)','\fontsize{14}ref. slope 1','Location','Northwest');
legend('boxoff');  

