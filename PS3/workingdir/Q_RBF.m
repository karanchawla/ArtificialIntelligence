%Modification of code by Dr. Chowdhary
function phi_sa = Q_RBF(s,a,params)

N_grid = params.N_grid;
N_s = params.N_s;
N_sa = params.N_sa;

centers = params.c;
% RBF variance
mu = params.mu;
% state-space slice approximation

phi_s = zeros(N_s,1);
phi_s(1)=params.bw;
for i = 1:N_s-1
    phi_s(1+i) = exp(-0.5*norm(s - centers(:,i))^2/mu(i));
end
%Output RBF phi_sa
phi_sa = zeros(N_sa,1);
phi_sa(((a-1)*N_s + 1):a*N_s) = phi_s;

end