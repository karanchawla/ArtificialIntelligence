%Author: Karan Chawla

clear all
close all

%% parameters of the linear system: x_dot=Ax+Bu; y= Cx
% Students: Try different linear systems, try higher dimensional linear systems

A=[-1 -5;6,-1];
B=[1;0];
x=[1,0]'; %initial value of the state vector
u=0; %initial input
C=[0 1];
y=C*x;

%%integration parameters
dt=0.01;
tf=10/dt;
t=0;

%This is a discrete implementation, so the following line makes the system discrete
% by taking the matrix exponential: Ad=expm(A*dt)
% so now our system becomes: x(k+1)=Ad*x_k+ Bd*u_k
Ad=expm(A*dt);


%noise parameters (you can change these and see how it affects the filter)
w=0.01;%process noise covariance
v=0.1;%measurement noise covariance
x_tilde=[0;0]; %This is where we initialize the state estimate, note how it is different from
               %the actual initial state (x=[1,0]), you can see how changing this affects the filter
 %just setting the inital values
%x_hat = x_tilde; %Using this for comparison purposes
x_hat=x; %More often than not - we know exactly what state the system is initialized in - so the filter
         %behaves better when we set the initial estimate of the system as
         %x instead of x_tilde
 
Q=eye(2)*w; %Q matrix captures the process noise covariance
R=eye(1)*v; % R matrix captuers the measurement noise covariance

P=eye(2)*1; % initial state error covariance matrix (P)

% storage variables
X_HAT_STORE= zeros(2,tf);
X_STORE= zeros(2,tf);
Y_REC = zeros(tf,1);
T_REC        = zeros(tf,1);

%% main loop
for k=1:tf
    %% system intergration
    % System Model 
    u=sin(0.01*k)*0.1; %The input to the system can be changed here
    x=Ad*x+B*u+randn(2,1)*w; %note the discrete propagation, and the addition of process noise
    y(k+1)=C*x+randn*v; %the measurement noise adds here
    
    t=t+dt;
    %% Add your code here
    %Predict state
    x_hat = Ad*x_hat + B*u;
    P = Ad*P*Ad'+ Q;
    
    %Update Step
    G = P*C'/(C*P*C' + R);
    x_hat = x_hat + G*(y(k+1) - C*x_hat);
    P = (eye(2) - G*C)*P;
    
    
    %% store
    X_HAT_STORE(:,k)= x_hat;
    X_STORE(:,k)=x;
    Y_REC(k)=y(k);
    T_REC (k) = t;
    
    
end

%% plotting

figure(1)
subplot(211)
plot(T_REC,X_STORE(1,:),T_REC,X_HAT_STORE(1,:))
subplot(212)
plot(T_REC,X_STORE(2,:),T_REC,X_HAT_STORE(2,:))

legend('real','estimated')
title('state estimate')

figure(2)
plot(T_REC,Y_REC,T_REC,X_HAT_STORE(2,:))
legend('real','estimated')
title('output')
