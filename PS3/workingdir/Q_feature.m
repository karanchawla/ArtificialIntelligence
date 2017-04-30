%modification of code by dr. chowdhary
function phi_sa=gridworld_Q_calculate_feature(s,a,params)
%call the appropriate feature function

if params.basis==0
    N_grid = params.N_grid;
    N_s = params.N_s;
    N_sa = params.N_sa;
    s_index = sub2ind([N_grid N_grid],s(1),s(2));
    phi_s = zeros(N_s,1);
    phi_s(s_index) = 1;
    phi_sa = zeros(N_sa,1);
    phi_sa(((a-1)*N_s + 1):a*N_s) = phi_s;
elseif params.basis==1
    phi_sa= Q_RBF(s,a,params);
end