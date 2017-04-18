function [x,xDot]=wingrock_sim(x,delta,dt,Wstar)


%% propogate state dynamics
clear xp
xp=state(x,delta,Wstar);
    xDot=xp;
   %1
%    rk1=dt*xp;
%    x1=x+rk1;
%    %2
%    xp=state(x1,delta,Wstar);
%    rk2=dt*xp;
%    x1=x+rk2;
%    %3
%    xp=state(x1,delta,Wstar);
%    rk3=dt*xp;
%    x1=x+2*rk3;
%    %4
%    xp=state(x1,delta,Wstar);
%    rk4=dt*xp;
%    
%    x=x+(rk1+2.0*(rk2+rk3)+rk4)/6;

x=x+dt*xp;
deltaErr=Wstar'*[1;x(1);x(2);abs(x(1))*x(2);abs(x(2))*x(2);x(1)^3];
% End main function   
% Begin nested functions
%% State model
    function [xDot] = state(x,delta,Wstar)
        x1Dot=x(2);
        deltaErr=Wstar'*[1;x(1);x(2);abs(x(1))*x(2);abs(x(2))*x(2);x(1)^3];

        x2Dot=delta+deltaErr;%delta-(deltaerr)
        xDot=[x1Dot;x2Dot];
        %Xdotdot=delta+sinx-mod(xdot)*xdot