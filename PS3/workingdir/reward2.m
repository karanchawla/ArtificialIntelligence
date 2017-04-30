%modification of code by dr. chowdhary
function [rew,breaker] = reward2(s_new,params)

s_goal = params.s_goal;
rew_goal = params.rew_goal;
rew_trans = params.rew_trans;

breaker = false; %stop if reached goal (terminate episode)

rew = 0;

% Check for goal
if((s_new(1) == s_goal(1))...
        && (s_new(2) == s_goal(2)))
    rew = rew_goal;
    breaker = true;
end
rew = rew+rew_trans;
end