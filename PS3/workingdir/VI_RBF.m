%Modification of code by Dr. Chowdhary
function phi = V_RBF(params)

N_grid = params.N_grid;
N_phi_s = params.s;
N_state = params.N_state;

centers = params.rbf_c;
% RBF variance
mu = params.mu;
% state-space slice approximation

phi = zeros(N_state,N_phi_s);
phi(:,1)=ones(N_state,1)*params.bw;

for s=1:N_state
    [sx sy]=ind2sub([N_grid N_grid],s);
    for i=1:(N_phi_s-1)
        phi(s,1+i) = exp(-0.5*norm([sx sy] - centers(i,:))^2/mu(i));
    end
end