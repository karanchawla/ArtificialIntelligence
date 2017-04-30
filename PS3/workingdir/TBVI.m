%% Gridworld Simulation
clear all;
close all;
clc;
%% Gridworld Parameters
basis               = 1;                % 0 for tabular, 1 for RBF
N_grid              = 5;                % size of grid
N_state             = N_grid*N_grid;    % size of state
N_act               = 5;                % number of actions
N_eps               = 20;               % max length of trajectory
N_time              = 1000;             % length of time
N_converge          = 6;                % number of convergences 
N_exec              = 5;                % number of executions
N_dim               = 2;                % state dimension

eta                 = 0.01;             % convergence tol
s_init              = [1;1];            % initial state
a_init              = 2;                % initial action
s_goal              = [N_grid;N_grid];  % goal state

rew_goal            = 1;                % goal reward
rew_trans           = -0.01;            % transition reward
noise               = 0.1;              % transition uncertainty

params.N_grid       = N_grid;           % size of grid
params.s_goal       = s_goal;           % goal state
params.rew_goal     = rew_goal;         % goal reward
params.rew_trans    = rew_trans;        % transition reward
params.N_dim        = N_dim;            % state dimension
params.N_act        = N_act;            % number of actions
params.noise        = noise;            % transition uncertainty

%% Learning Parameters
gamma               = 0.9;              % discount factor
params.gamma        = gamma;            % discount factor

alpha_init          = 0.5;              % initial learning rate
alpha_dec           = 0.5;              % decay rate
mu                  = 1;
eps_init            = 0.8;              % initial exploration rate
eps_dec             = 0.1;              % exploration rate decay

max_points          = 25;               % max centres allowed for RBF
tol                 = 1e-4;             % tolerance

if basis==0
    params.N_s = N_state;                   % Number state-features
    params.N_sa = params.N_s*params.N_act;  % Number state-action-features
    params.state_action_slicing_on=1;
    gpr = onlineGP_RL(0,0,0,0,params);
    params.basis=0;
elseif basis==1
    params.c = [5 5; 1 5; 5 1; 1 1]';
    params.N_s = size(params.c,2)+1;        % Number state-features
    rbf_mu = ones(params.N_s,1)*mu;         % RBF mu
    params.mu=ones(params.N_s,1)*1;         % RBF mu
    params.bw=1;                            % RBF bias
    params.N_sa = params.N_s*params.N_act;  % Number state-action-features
    params.state_action_slicing_on=1;
    gpr = onlineGP_RL(0,0,0,0,params);
    params.basis=1;
end

%% Algorithm Execution
%Create state transition matrix
P=zeros(N_act,N_state,N_state);
for s=1:N_state
    %states: stay, right, up, left, down
    new_s(1) = s;                               % stay
    new_s(2) = s+1*(mod(s,N_grid)~=0);          % right
    new_s(3) = s+N_grid*(s+N_grid<=N_state);    % up
    new_s(4) = s-1*(mod(s,N_grid)~=1);          % left
    new_s(5) = s-N_grid*(s-N_grid>=1);          % down
    
    % for all actions
    for a=1:N_act
        % compute probability of desired transition
        P(s,new_s(a),a) = (1-noise);
        % for all actions
        for a_rand=1:N_act
            % compute probability of uncertain transition
            P(s,new_s(a_rand),a) = P(s,new_s(a_rand),a) + noise/N_act;
        end
    end
end

% initialize
theta = zeros(params.N_sa,1);
ctr = 0;
eval_ctr = 0;
n_conv=0;
i=0;

% while the time is less than the required time
% while the number of convergences is less that that required
while i<=N_time && n_conv<N_converge
    % initialize
    s_old = s_init;
    break_cmd = 0;
    delta = 0;
    k=1;
    % while less than the number of episodes
    % while not commanded to break
    while k<=N_eps && ~break_cmd
        ctr = ctr+1;

        % set the exploration proability
        p_eps = eps_init/(ctr)^eps_dec;
        
        % check whether to explore
        r = sample_discrete([p_eps 1-p_eps]);
        
        % explpre
        if r==1
            p = 1/N_act.*ones(1,N_act);
            action = sample_discrete(p);
        % exploit
        else
            [Q_opt,action] = Q_greedy_act(theta,s_old,params,gpr);
        end

        % compute next state and reward
        s_old_lin = sub2ind([N_grid N_grid],s_old(1),s_old(2));
        s_new=zeros(2,N_exec);
        rew=zeros(1,N_exec);
        % for all executions
        for j=1:N_exec
            % calculate next state
            s_new(1,j) = sample_discrete(P(s_old_lin,:,action));
            [s_new(1,j),s_new(2,j)]=ind2sub([N_grid N_grid],s_new(1,j));
            % calculate reward
            [rew(j),break_cmd] = reward2(s_new(:,j),params);
        end
        % recompute learning rate
        alpha =  alpha_init/ctr^alpha_dec;
        % recompute feature vector
        phi_old = Q_feature(s_old,action,params);
        % calculate the value
        val_old = Q_value(theta,s_old,action,params);
        v_new = 0;
        % for all executions
        for j=1:N_exec
            % compute optimal action
            [Q_opt,a_op] = Q_greedy_act(theta,s_new(:,j),params,gpr);
            % compute new value
            v_new=v_new+rew(j)+gamma*Q_value(theta,s_new(:,j),a_op,params);
        end
        v_new = v_new/N_exec;
        % compute error
        err = (v_new - val_old);
        
        % update basis vector
        theta_old=theta;
        theta = theta + alpha*(err.*phi_old);
        delta = max(delta,abs(max(theta-theta_old)));
        
        % reset state
        s_old = s_new(:,1);
        k=k+1;
    end
    % check for break condition
    if (delta<eta && break_cmd)
        n_conv = n_conv+1;
    end
    i=i+1;
end

% update policy
policy=zeros(N_grid,N_grid);
for i=1:N_grid
    for j=1:N_grid
        [~,policy(i,j)] = Q_greedy_act(theta,[i;j],params,gpr);
    end
    policy(N_grid,N_grid)=1;
end
policy = flipud(policy');

%% Monte Carlo runs
rew_eval = zeros(1,100);
for eval_ctr = 1:100
    
    s_old = s_init;
    for ctr=1:N_state
        
        [~,action] = Q_greedy_act(theta,s_old,params,gpr);
        s_next = gridworld_trans(s_old,action,params);
        [rew,break_cmd] = reward2(s_next,params);
        rew_eval(eval_ctr) = rew_eval(eval_ctr) + rew;
        
        if break_cmd
            break;
        end
        
        s_old = s_next;
    end
end
boxplot(rew_eval);
rew_mean=mean(rew_eval);
rew_var=var(rew_eval);
display(rew_mean);
display(rew_var);