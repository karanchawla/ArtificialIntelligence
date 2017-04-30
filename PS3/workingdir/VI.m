clear all; close all; clc;

%Initialize Variables
basis          = 1;                             % basis
N_grid         = 5;                             % grid size
N_state        = N_grid*N_grid;                 % state size
N_act          = 5;                             % action size
noise          = 0.1;                           % transition uncertainty
gamma          = 0.9;                           % discount factor
eta            = 0.05;                          % tolerance
alpha0         = 0.5;                           % initial Learning Rate
alpha_exp      = 0.5;                           % learning rate 

% create value function tablse and policy
v              = zeros(N_state, 1);             % value function
old_v          = zeros(N_state, 1);             % previous value function
policy         = zeros(N_state,1);              % policy

%Create state transition matrix
P              = zeros(N_state,N_state,N_act);  %(A, Sold, Snew)

params.N_grid  = N_grid;
params.N_state = N_state;
params.rbf_c   = [1 1; 1 5; 2 4; 3 3; 4 2; 5 1; 5 5];
params.s       = size(params.rbf_c,1)+1;        % RBF standard deviation
params.mu      = ones(params.s,1)*1;            % RBF average
params.bw      = 1;                             % RBF bias parmeter

for s=1:N_state
    %states: stay, right, up, left, down
    new_s(1)   = s;                             % stay
    new_s(2)   = s+1*(mod(s,N_grid)~=0);        % right
    new_s(3)   = s+N_grid*(s+N_grid<=N_state);  % up
    new_s(4)   = s-1*(mod(s,N_grid)~=1);        % left
    new_s(5)   = s-N_grid*(s-N_grid>=1);        % down
    
    % for all actions
    for a=1:N_act
        % compute transition probability for desired
        P(s,new_s(a),a) = (1-noise);
        % for all actions
        for a_rand=1:N_act
            % compute uncertain transition probability
            P(s,new_s(a_rand),a) = P(s,new_s(a_rand),a) + noise/(N_act-1);
        end
    end
end

% choose basis


% tabular
if basis==0
    % initialize delta
    delta=eta;
    % while delta is not within the tolerance
    while delta>=eta
        % initialize delta
        delta=0;
        % create a temporary set of value and policy vectors
        va = zeros(N_state, N_act);
        v_ = zeros(N_state,1);
        %for all initial states
        for s=1:N_state
            % for all actions
            for a=1:N_act
                % for all final states
                for s_prime=1:N_state
                    % compute the new value function
                    va(s,a)=va(s,a)+P(s,s_prime,a)*(reward(s_prime)+gamma*v(s_prime,:));
                end
            end
            % store the new maximum value and corresponding action
            [v_(s,:), action] = max(va(s,:));
            policy(s,:) = action;
            old_v(s,:) = v(s,:);
            % update the value function and recompute delta
            v(s,:) = v_(s,:);
            delta = max(delta,abs(v(s,:)-old_v(s,:)));
        end
    end
    
elseif basis==1 % RBF
    % initialize theta, phi, and delta
    theta = zeros(params.s,1);
    phi = VI_RBF(params);
    delta=eta;
    i=1;
    while delta>=eta
        delta=0;
        for s=1:N_state
            v=phi(s,:)*theta;
            %find action that maximizes value
            va=zeros(1,N_act);
            for a=1:N_act
                va(a)=P(s,:,a)*(reward(1:N_state)'+gamma*phi*theta);
            end
            v_=max(va);
            % update RBF
            alpha = alpha0/i^alpha_exp;
            theta_old = theta;
            theta = theta + alpha*(v_-v)*phi(s,:)';
            theta=theta/norm(theta);
            % update delta
            delta = max(delta,max(theta-theta_old));
        end
        i=i+1;
    end
    
    % update value function using RBF
    v = phi*theta;
    Va=zeros(N_state,N_act);
    %compute maximum value
    for a=1:N_act
        Va(:,a)=P(:,:,a)*v;
    end 
    %update policy;
    [~, policy]=max(Va,[],2);
end

% display output
V_new=flipud(reshape(v,[N_grid N_grid])')/max(v);
policy=flipud(reshape(policy,[N_grid N_grid])');
display(V_new)
display(policy)