%% Gridworld Simulation
clear all;
close all;
clc;
%% Gridworld Parameters
basis               = 1;                    % 0 for tabular, 1 for RBF
N_grid              = 5;                    % grid size
N_state             = N_grid*N_grid;        % state size
N_act               = 5;                    % number of actions

N_dim               = 2;                    % state dimension
s_init              = [1;1];                % initial state
a_init              = 2;                    % initial action
s_goal              = [N_grid;N_grid];      % goal state

N_obstacle          = 0;                    % number of obstacles
obs_list            = [];                   % coordinates of obstacles

rew_goal            = 1;                    % reward for goal
rew_trans           = -0.01;                % reward for transition
noise               = 0.1;                  % transition uncertainty

params.N_grid       = N_grid;               % grid size
params.s_goal       = s_goal;               % goal state
params.rew_goal     = rew_goal;             % reward for goal
params.rew_trans    = rew_trans;            % reward for transition
params.N_dim        = N_dim;                % state dimension
params.N_act        = N_act;                % number of actions
params.noise        = noise;                % transition uncertainty

%% Learning Parameters
N_length            = 100;                  % length of episode
N_eps               = 200;                  % number of episodes
N_exec              = 3;                    % number of executions
N_freq              = 100;                  % frequency of  evaluation
N_eval              = 30;                   % evaluation interations
N_budget            = 25;                   % max points in stack

gamma               = 0.9;                  % discount factor
params.gamma        = gamma;                % discount factor

params.N_budget     = N_budget;             % max points in stack
data_method         = 2;                    % 1 For cylic and 2 for SVD
params.epsilon_data_select=0.2;
stack_index         = 0;                    % initial stack index         
points_in_stack     = 0;                    % initial stack length

alpha_init          = 0.5;                  % initial learning rate
alpha_dec           = 0.5;                  % learning rate decay
eps_init            = 0.8;                  % initial exploration rate
eps_dec             = 0.1;                  % exploration rate decay
N_pts               = 25;                   % max centres allowed for RBF
tol                 = 1e-4;                 % tolerance

if basis==0
    params.N_s = N_state;                   % state-features  
    params.N_sa = params.N_s*params.N_act;  % state-action-features 
    params.basis=basis;
    params.state_action_slicing_on=1;
    params.basis=0;
    gpr = onlineGP_RL(0,0,0,0,params);
elseif basis==1
    params.c = [5 5; 1 5; 5 1; 1 1]';
    params.N_s = size(params.c,2)+1;        % number of state-features 
    params.mu=ones(params.N_s,1)*1;         % RBF mu
    params.bw=1;                            % RBF bias
    params.N_sa = params.N_s*params.N_act;  % state-action features
    params.state_action_slicing_on = 1;
    params.basis = 1;
    gpr = onlineGP_RL(0,0,0,0,params);
end

%% Algorithm Execution
rew_exec = zeros(N_exec,1);
eval_ctr = zeros(N_exec,1);

% for all executions
for i =1:N_exec
    % reset Q function
    theta = zeros(params.N_sa,1); 
    
    step_ctr = 1;
    % for all episodes
    for j = 1:N_eps
        %fprintf('Episode: %d/%d, Execution: %d/%d \n',j,N_eps,i,N_exec);
        
        % reset to initial state
        s_old = s_init;
        
        % set exploration probability
        p_eps = eps_init/(step_ctr)^eps_dec;
        % check if exploring
        r = sample_discrete([p_eps 1-p_eps]);
        % explore
        if r==1 
            p = 1/N_act.*ones(1,N_act);
            action = sample_discrete(p);
        % exploit
        else 
            [Q_op,action] = Q_greedy_act(theta,s_old,params,gpr);
        end
        % for lenfth of evaluation
        for k = 1: N_length
            % check if it is time to evaluate
            if(mod(step_ctr,N_freq) == 0)
                % evaluate reward
                eval_ctr(i) = eval_ctr(i) + 1;
                rew_eval = zeros(1,N_eval);
                % for number of evals
                for eval_count = 1:N_eval
                    % reset state
                    s_prv = s_init;
                    % for legnth of evaluation
                    for step_count = 1:N_length
                        % calculate optimal action
                        [Q_op,action]=Q_greedy_act(theta,s_prv,params,gpr);
                        s_next = gridworld_trans(s_prv,action,params);
                        % calculate reward
                        [rew, break_cmd] = reward2(s_next,params);
                        rew_eval(eval_count) = rew_eval(eval_count)+rew;
                        % check break condition
                        if break_cmd
                            break;
                        end
                        % update state
                        s_prv = s_next;
                    end
                end
                % update reward
                rew_exec(i,eval_ctr(i)) = mean(rew_eval);
            end  
            step_ctr = step_ctr + 1;
            
            % get nextsState
            s_new = gridworld_trans(s_old,action,params);
            
            % calculate reward
            [rew,break_cmd] = reward2(s_new,params);
            
            % set exploration rate
            p_eps = eps_init/(step_ctr)^eps_dec;
            % check if exploring
            r = sample_discrete([p_eps 1-p_eps]);
            % explore
            if r==1 
                p = 1/N_act.*ones(1,N_act);
                a_new = sample_discrete(p);
            % exploit
            else 
                % calculation optimal action
                [Q_op,a_new] = Q_greedy_act(theta,s_old,params,gpr);
            end
            
            % calculate learning rate
            alpha =  alpha_init/(step_ctr)^alpha_dec;
            % calculate feature vector
            phi_old = Q_feature(s_old,action,params);
            % calculate value
            v_old = Q_value(theta,s_old,action,params);
            % update value
            v_new = Q_value(theta,s_new,a_new,params);
            % compute error
            err = (rew + gamma*v_new - v_old);
            % update RBF parameter
            theta = theta + alpha*(err.*phi_old);
            
            % reset state
            s_old = s_new;
            action = a_new;
            
            % check break comdition
            if break_cmd
                break;
            end
        end
    end
end

%% Post Process
% find  minimum number of evaluations
min_eval = min(eval_ctr);
rew_exec = rew_exec(:,1:min_eval);
rew_total = zeros(1,min_eval);
std_total = zeros(1,min_eval);

% update reward
for m =1:min_eval
    rew_total(m) = mean(rew_exec(:,m));
    std = var(rew_exec(:,m));
    std_total(m) = 0.1*std;
end

%% Plots
t = 1:min_eval;
t = t.*N_freq;
errorbar(t,rew_total,std_total);
xlabel('episodes')
ylabel('reward')