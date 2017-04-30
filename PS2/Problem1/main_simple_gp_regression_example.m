%% simple GP regression example
%author: Girish Chowdhary, 
%using publically available code on 
%http://web.mit.edu/girishc/www/resources/resources%20files/Module_4_Nonparameteric_Adaptive_Control.zip
% Thanks to Hassan Kingravi for writing a lot of the onlineGP function

clear all
close all

%% load input data
load('data.mat', 'data') % gp_regression_example_data
% the only data we want for this example is 2 inputs X1_IN, X2_IN, and an
% output Y1_OUT. Example data is loaded by the above program

%% invoke GP regression model object
%% Gaussian Process parameter settings and initialization
%parameters
bandwidth = 100; %this is the bandwidth sigma of the square exponential 
                 %kernel we are using, the kernel is given by
% k(x1,x2)=exp(-norm(x1-x2)^2/bandwidth^2
noise = 1; % this is the noise we assume that our data has, right now its 
           %set to 1, this needs to be inferred from data, there are
           %data driven techniques available to do this 
           %(see Rassmussen and Williams 2006 )
tol = 0.00001;% this is a parameter of the sparsification process, 
                %smaller results in the algorithm picking more kernels 
                %(less sparse)
                %This has been explained in Chowdhary et al. ACC 2012
                %(submitted)
max_points=100;%this is our budget of the kernels, we allow 100 here, with 
                %real data this number needs to be tuned, 
                %althoug the regression is not too sensitive to it with
                %appropriate bandwidth its mostly for limiting 
                %computationl effort

%the following line invokes the gp regression class
gpr = onlineGP(bandwidth,noise,max_points,tol);
%the goal is to learn a generative model from data
%let the mean function be f(x_in), the measuremente are y, and they are
% y =f(x_in)+noise_function(noise), currently our model of noise is white
% noise
% The KL diverence based sparsification method developed by Csato et 
% al. is used, and is referenced in the class
%the GPR function has several subfunctions, which are called as
%initialization function: gpr.process(x_in,y)
%regression/learning function gpr.update(x_in,y)
%prediction function gpr.predict(x_in)
%model saving function gpr.save('model_name');
%model loading function gpr.load('model_name')
%get internal variables: gpr.get('var_name'), var_names are documented 
% in the function object itself, the main ones are:
% The GP current basis: 'basis' or 'BV'
% Current set of active observations: 'obs'
% Current set of kernels: 'K','kernel'
% Current size of active basis set: 'current_size' or 'size' or
%'current size'
% see more definitions in the class itself
THETA=data(:, 1);
P=data(:, 2);
U=data(:, 3);
PDOT=data(:, 4);

%% loop through data to learn
for ii=1:max(size(PDOT))
    x_in=[THETA(ii);P(ii);U(ii)];
    if ii == 1
        % if first step, initialize GP
        gpr.process(x_in,PDOT(ii));
    else
        gpr.update(x_in,PDOT(ii));
    end
end

%% now we can predict
load('datanew.mat', 'data')
THETA_new=data(:, 1);
P_new=data(:, 2);
U_new=data(:, 3);
PDOT_new=data(:, 4);

%define the grid over which we are going to predict
range_size=max(size(P_new));

%%
%loop through the grid to get the predicted values NEW DATA
for ii=1:range_size
        x_in=[THETA_new(ii);P_new(ii);U_new(ii)];
        [mean_post, var_post] = gpr.predict(x_in);
        EST_MEAN_POST_GP(ii)=mean_post;
        EST_VAR_POST_GP(ii)=var_post;
end
MSE=(PDOT_new-EST_MEAN_POST_GP');
MSE=norm(MSE, 2)/range_size;