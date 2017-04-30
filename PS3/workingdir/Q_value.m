%modification of code by dr. chowdhary
function val = Q_value(theta,s,a,params,gpr)

if params.basis==0
    N_grid = params.N_grid;
    N_s = params.N_s;
    N_sa = params.N_sa;
    s_index = sub2ind([N_grid N_grid],s(1),s(2));
    phi_s = zeros(N_s,1);
    phi_s(s_index) = 1;
    phi_sa = zeros(N_sa,1);
    phi_sa(((a-1)*N_s + 1):a*N_s) = phi_s;

    val = phi_sa'*theta;
elseif params.basis==1
    phi_sa= Q_RBF(s,a,params);
    val = theta'*phi_sa;
elseif params.basis==3 
    x=[s;a];
    [mean_post var_post] = gpr.predict(x);      
     val = mean_post;
end
end