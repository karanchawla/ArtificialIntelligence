%code by dr. chowdhary
function [Q_val_opt,action] = Q_greedy_act(theta,s,params,gpr)
N_act = params.N_act;
act_val = zeros(1,N_act);
for a=1:N_act                
          act_val(a) = Q_value(theta,s,a,params,gpr);
end
[Q_val_opt,action] = max(act_val);
end