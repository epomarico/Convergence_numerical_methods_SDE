% STRONG CONVERGENCE OF THE EULER-MARUYAMA AND MILSTEIN NUMERICAL METHODS
% FOR A LINEAR SDE
%
% This program tests the strong convergence of the Euler-Maruyama (EM) algorithm,
% the Milstein (M) one, as well as its derivative-free (M2), for the numerical 
% solution of the following stochastic differential equation
%
% dX = lambda*X dt + mu*X*dW, X(0) = Xzero,
% with lambda,mu constants and Xzero = 1.
%
% To study the strong convergence of the methods we calculate 
% the error E(|X_T - X(T)|) which is the sample average of the absolute value 
% of the differences between the numerical solution and the exact solution
% at time T X_T and X(T).

clear all

% Initialize random number generator
randn('state',100)

% Parameters of the equation
lambda = -2; mu = 1.0; Xzero = 1; 
% Time parameters
T = 1; N = 2^9; dt = T/N;  Dtvals = dt*(2.^([0:4]));     
% Number of brownian trajectories
M = 1000;                     
% Function g(X) which multiples dW
g=@(mu,x) mu*x;
dx=0.01; %spatial step for derivative calculation

% Initialize matrices for errors for the three algorithms
XerrEM = zeros(M,5);             
XerrM = zeros(M,5);  
XerrM2 = zeros(M,5);   

for s = 1:M  % same calculation for each brownian trajectory  
    dW = sqrt(dt)*randn(1,N);  % Brownian increments
    W = cumsum(dW);            % Brownian trajctory
    Xtrue = Xzero*exp((lambda-0.5*mu^2)*T +mu*W(end));  % Exact solution for X(T) at the time T
    for p = 1:5                 % consider 5 different time-steps 
        R = 2^(p-1); Dt = R*dt; L = N/R; % set the value of the time-step Dt and the number of steps to consider
                                         % over which we run the numerical calculation 
        XtEM = Xzero; XtM = Xzero; XtM2 = Xzero; % initialize the variable X(T) for the three algorithms 
        for j = 1:L
            Winc = sum(dW(R*(j-1)+1:R*j)); % Brownian increment calculated over the chosen step-size
            XtEM = XtEM + Dt*lambda*XtEM + mu*XtEM*Winc; %Euler - Maruyama method
            XtM = XtM + Dt*lambda*XtM + mu*XtM*Winc + 0.5*mu^2*XtM*(Winc^2-Dt); %Milstein method
            
            gp=(1/(2*dx))*(g(mu,XtM2+dx)-g(mu,XtM2-dx)); %Calculation of the derivative of g(X)
            XtM2 = XtM2 + Dt*lambda*XtM2 +g(mu,XtM2)*Winc + 0.5*gp*g(mu,XtM2)*(Winc^2-Dt); % Derivative-free Milstein method
        end
        XerrEM(s,p) = abs(XtEM - Xtrue); % Error of EM method at t=T
        XerrM(s,p) = abs(XtM - Xtrue);   % Error of M method at t=T
        XerrM2(s,p) = abs(XtM2 - Xtrue); % Error of M2 method at t=T
    end
end

%Fitting of the errors for the three algorithms
EM=polyfit(log10(Dtvals),log10(mean(XerrEM)),1);
slope_EM=EM(1)
Mil=polyfit(log10(Dtvals),log10(mean(XerrM)),1);
slope_Mil=Mil(1)
Mil2=polyfit(log10(Dtvals),log10(mean(XerrM2)),1);
slope_Mil2=Mil2(1)

% Plotting the graph of strong convergence for the three methods 

close all
str = {'\fontsize{14}dX = \lambda Xdt + \muXdW','X(0)=1,\lambda = -2,\mu=1, M=10^3'};

% EM 
figure;
subplot(1,3,1)
loglog(Dtvals,mean(XerrEM),'b*-'), hold on
loglog(Dtvals,Dtvals.^0.5,'r--'), hold off % reference line with slope 0.5
xlabel('\fontsize{16}\Delta t'), ylabel('\fontsize{16}E(|X(T) - X_N|)')
title('\fontsize{16}Strong convergence','FontSize',10) 
dim = [0.19 0.03 0.26 0.25];
annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','none');
dim1 = [0.09 0.74 0.26 0.25];
annotation('textbox',dim1,'String','\fontsize{16}a)','FitBoxToText','on','EdgeColor','none');
legend('\fontsize{16}Euler-Maruyama','\fontsize{14}reference slope 1/2','Location','Northeast')
legend('boxoff')  

% M
subplot(1,3,2)
loglog(Dtvals,mean(XerrM),'b*-'), hold on
loglog(Dtvals,Dtvals,'r--'), hold off % reference line with slope 1
xlabel('\fontsize{16}\Delta t'), ylabel('\fontsize{16}E(|X(T) - X_N|)')
title('\fontsize{16}Strong convergence','FontSize',10)
dim = [0.47 0.03 0.26 0.25];
annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','none');
dim2 = [0.37 0.74 0.26 0.25];
annotation('textbox',dim2,'String','\fontsize{16}b)','FitBoxToText','on','EdgeColor','none');
legend('\fontsize{16}Milstein','\fontsize{14}reference slope 1','Location','Northeast')
legend('boxoff')  

% M2
subplot(1,3,3)
loglog(Dtvals,mean(XerrM2),'b*-'), hold on
loglog(Dtvals,Dtvals,'r--'), hold off % reference line with slope 1
xlabel('\fontsize{16}\Delta t'), ylabel('\fontsize{16}E(|X(T) - X_N|)')
title('\fontsize{16}Strong convergence','FontSize',10)
dim = [0.75 0.03 0.26 0.25];
annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','none');
dim3 = [0.65 0.74 0.26 0.25];
annotation('textbox',dim3,'String','\fontsize{16}c)','FitBoxToText','on','EdgeColor','none');
legend('\fontsize{16}Milstein (derivative-free)','\fontsize{14}reference slope 1','Location','Northeast')
legend('boxoff')  



