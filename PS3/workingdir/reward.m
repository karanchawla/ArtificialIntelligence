function r=reward(s)
    goal_s=25;
    goal_rew=1;
    trans_rew=-0.01;
    r = goal_rew*(s==goal_s) + trans_rew*(s~=goal_s);
end