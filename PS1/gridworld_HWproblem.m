%% Gridworld Simulation with GPs for Q learning
clear all;
close all;
clc;
%% Gridworld Parameters
N_grid = 5; % Number of grids on one direction
N_state = N_grid*N_grid; % Square Gridworld
N_act = 5; % 4 directions + null action

n_state_dim=2;
s_init = [1;1]; % Start in the top left corner
a_init=2;
s_goal = [N_grid;N_grid]; % Goal is at the upper right corner [end,end]
N_eps_length =100; % Length of an episode

%% storage variables
STATE_STORE = zeros(N_eps_length,2);
%start node
start.x = s_init(1)
start.y = s_init(2)
start.g = 0
x = s_init(1);
y = s_init(2);
g = 0; %cost

open(1,:) = [start];

found = 0; %flag that is set when search is complete
resign = 0; %flag set if we can't expand

visited = zeros(N_grid);
action = zeros(N_grid);

for i=1:N_grid
    for j=1:N_grid
        visited(i,j)=0;
        action(i,j)=-1;
    end
end
%% Algorithm Execution
        s_old = s_init;
%         for k = 1: N_eps_length            
%             %% This is the main part of the code
%             % As the code stands, it computes the next state given an
%             % action for N_eps_length iterations, 
%             % there are 5 actions, they are described in
%             % gridworld_trans_HW
%             % the purpose of this code is to show you how to use the
%             % gridworld_trans_HW function and plot the results
%             % in the current code the action has been fixed
%             % your job is to write a search algorithm that gets you from
%             % s_init to s_goal

        while(not(found) && not(resign))
            if isempty(open)
                resign = 1;
            else
         
                [val,idx] = min(open);
                next = open(idx);
                open = open(2:end);
                x = next(1);
                y = next(2);
                g = next(3);
                
                if(x==s_goal(1) && y ==s_goal(2))
                    found = 1;
                else
                    for i=2:5
                        s_new = gridworld_trans_HW(s_old,action,N_grid); 
                        if(s_new(1)>=0 && s_new(1)<=N_grid && s_new(2)>=0 && s_new(2)<=N_grid) 
                            if(closed(s_new(1),s_new(2))==0)
                                g2 = g + cost;
                                cat(open,[g2,s_new(1),s_new(2)]);
                                closed(s_new(1),s_new(2))=1;
                                count = count + 1;
                            end
                        end
                    end
                end
            end
        end
                        

%% Plots
figure(1)
plot(STATE_STORE(:,1),STATE_STORE(:,2),'o')
grid on
xlabel('x state')
ylabel('y state')