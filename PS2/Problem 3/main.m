%Author: Karan Chawla
%11th March '17

%initialization
clc;
clearvars;
close all;

%setup the parameters to be used 
load ('data.mat' , 'data');
theta = data(:,1);
p = data(:,2);
u = data(:,3);
p_dot = data(:,4)';

X(1,:) = theta;
X(2,:) = p;
X(3,:) = u;
y = p_dot;

input_layers = 3;
hidden_layers = 16;

%normalize the data set to have zero mean
% [X,mu,sigma] = featureNormalize(X);
% [y,mu2,sigma2] = featureNormalize(y);

X(1,:) = X(1,:)/max(abs(X(1,:)));
X(2,:) = X(2,:)/max(abs(X(2,:)));
X(3,:) = X(3,:)/max(abs(X(3,:)));
y = y + ones*abs(min(y));
y = y/max(abs(y));

%Load test data
load('datanew.mat','data');
thetav = data(:,1);
pv = data(:,2);
uv = data(:,3);
p_dotv = data(:,4);

XV(1,:) = thetav;
XV(2,:) = pv;
XV(3,:) = uv;
yv = p_dotv;

XV(1,:) = XV(1,:)/max(abs(XV(1,:)));
XV(2,:) = XV(2,:)/max(abs(XV(2,:)));
XV(3,:) = XV(3,:)/max(abs(XV(3,:)));
yv = yv + ones*abs(min(yv));
yv = yv/max(abs(yv));

%normalize the data set to have zero mean
% [XV,muv,sigmav] = featureNormalize(XV);
% [yv,muv2,sigmav2] = featureNormalize(yv);

epochs = 100;
a1 = zeros(hidden_layers,1);
v1 = zeros(hidden_layers,1);

Theta1(1,:) = 0.01*rand(1,hidden_layers);
Theta1(2,:) = 0.02*rand(1,hidden_layers);
Theta1(3,:) = 0.01*rand(1,hidden_layers);
Theta2 = 0.1*rand(1,hidden_layers);

eta = 0.1;%Learning rate

erms = zeros(epochs,1);
ermsTest = zeros(epochs,1);

for i = 1:epochs
    e = 0;
    for j = 1:size(X,2)
        a2 = 0;
        % Calculation of output (Training data)
        for k=1:hidden_layers
            a1(k) = X(1,j)*Theta1(1,k) + X(2,j)*Theta1(2,k) + X(3,j)*Theta1(3,k);
            v1(k) = sigmoid(a1(k));
            a2 = a2 + v1(k)*Theta2(1,k);
        end
        out = a2;
        e = e + computeCost(out,y(j));
        
        for k = 1:hidden_layers
          [Theta2, Theta1] = updateWeights(X, y, out, Theta1, Theta2, k, j, eta, v1);
        end
    end
    erms(i) = sqrt(e/length(p_dot));
    
    e_test = 0;
    
    for j = 1:length(p_dotv)
        a2 = 0;
        for k = 1:hidden_layers
            a1(k) = X(1,j)*Theta1(1,k) + X(2,j)*Theta1(2,k) + X(3,j)*Theta1(3,k);
            v1(k) = sigmoid(a1(k));
            a2 = a2 + v1(k)*Theta2(1,k);
        end
            out_test(j) = a2;
            e_test = e_test + computeCost(out_test(j),yv(j));
    end
    ermsTest(i) = sqrt(e_test/length(p_dotv));            
end

