%gaussian mixture model clusturing with EM algo
% see Tomasi's notes

clear all
close all
%%
load gauss_mix_data

k=max(size(means));% parameteric method, so number of parameters known
N=max(size(X));
prior_mean=10*randn(2,5);
prior_var=vars*0+1;
rho=[0.8 0.1 0.05 0.025 0.025];
rho_post=rho;

posterior_mean=prior_mean;
posterior_var=prior_var;

epochs=100;

%% your code here
for k=1:20
    for i=1:epochs
        for j=1:5
            sum(j,:)= summation(X,Y,prior_mean,j);
        end
        %assign data points to clusters
        posterior_mean = computeCentroids(sum,X,Y);
        prior_mean = posterior_mean;
        
    end
    posterior_mean_TOT_1(k,:) = posterior_mean(1,:);
    posterior_mean_TOT_2(k,:) = posterior_mean(2,:);
end
figure(1)
histogram(posterior_mean_TOT_1,'BinMethod','integers');
figure(2)
histogram(posterior_mean_TOT_2,'BinMethod','integers');

%%
figure(3)
plot(x(:,1),y(:,1),'or')
hold on
plot(x(:,2),y(:,2),'*g')
hold on
plot(x(:,3),y(:,3),'+m')
hold on
plot(x(:,4),y(:,4),'.k')
hold on
plot(x(:,5),y(:,5),'+y')
%% this plots the posterior mean and variance
% if your inference works, you should be able to see the right allocation
for kk=1:5
    plot_gaussian_ellipsoid(posterior_mean(:,kk),diag([posterior_var(kk),posterior_var(kk)]));
end
%plot(posterior_mean(1,:),posterior_mean(2,:),'*')