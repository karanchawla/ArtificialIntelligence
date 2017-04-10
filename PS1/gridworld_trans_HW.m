function s_new = gridworld_trans_HW(s_old,action,N_grid)

% function s_new = gridworld_trans(s_old,action,params)
% returns the next state in s_new given
% old state in s_old and action in action, actions take on 5 values:
% 1: null action, stay where you are
% 2: go right
% 3: go up
% 4: go right
% 5: go down
% parameter N_grid contains the number of grids in the grid world


%always step in the action intended   
dir = action;
    
             % Get The Next State
            
            s_new = zeros(2,1);
            
            switch dir
               
                case 1 % Null
                    s_new(1) = s_old(1); 
                    s_new(2) = s_old(2);
                    
                case 2 % Right
                    s_new(1) = s_old(1) + 1; 
                    s_new(2) = s_old(2);
                    
                case 3 % Up
                    s_new(1) = s_old(1); 
                    s_new(2) = s_old(2) + 1;
                    
                case 4 % Left
                    s_new(1) = s_old(1) - 1; 
                    s_new(2) = s_old(2);
                    
                case 5 % Down
                    s_new(1) = s_old(1); 
                    s_new(2) = s_old(2) - 1;
                                       
                    
            end
            
            
         
         % Saturate the states if on boundaries
            
            s_new(1) = max([1,min([N_grid,s_new(1)])]);
            s_new(2) = max([1,min([N_grid,s_new(2)])]);
                      

end