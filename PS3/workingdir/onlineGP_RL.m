%============================== onlineGP ==================================
%  This code implements the sparse online GP algorithm presented in the
%  reference for basic GP regression with a Gaussian kernel.
%
%  This code is currently designed strictly for Gaussin kernels: if
%  you wish to extend it for non-Gaussian kernels, you MUST change
%  the values of k* away from unity!
%
%  Reference(s):
%    Sparse Online Gaussian Processes -Csato and Opper, Tech Report
%    Csato's thesis
%
%  Inputs:
%    sigma  	 - bandwidth for the Gaussian kernel; either
%                  1 x 1 scalar or
%                  1 x d vector
%    noise      -  parameter for noise in GP; assumed given by oracle
%    m          -  the size of your budget
%    tol        -  tolerance for projection residual
%
%  Outputs:
%    see functions
%============================== onlineGP ==================================
%  Name:		onlineGP.m
%  Authors: 		Hassan A. Kingravi, Girish Chowdhary
%============================== onlineGP ==================================
function oGP = onlineGP_RL(sigma,noise,max_points,tol,params)
if params.state_action_slicing_on==0
    
    BV           = [];            % Basis vector set
    K            = [];            % Kernel matrix
    alpha        = [];            % mean parameter
    C            = [];            % inverted covariance matrix
    Q            = [];            % inverted Gram matrix
    current_size = [];
    obs          = [];
    rew          = [];
    index        = 1;
elseif params.state_action_slicing_on==1
    BV_store           = zeros(params.N_dim,max_points*params.N_act);
    K_store            = zeros(max_points,max_points*params.N_act);
    alpha_store        = zeros(params.N_act,max_points);
    C_store            = K_store;
    Q_store            = K_store;
    current_size_store = ones(1,params.N_act);% min size is 1
    obs_store          = zeros(params.N_act,max_points);
    rew_store          = zeros(params.N_act,max_points);
    index_store        = ones(1,params.N_act);
end

oGP.process = @process;
oGP.predict = @predict;
oGP.update  = @update;
oGP.get = @oGP_get;

%------------------------------- process -------------------------------
%  Takes in a collection of data and generates an initial Gaussian
%  process model with the associated kernel matrix, its inversion and
%  the alpha vector.
%
%  Inputs:
%    data  	 - d x n data matrix passed in columnwise
%    y         - 1 x n column vector of observations
    function process(data,reward,Qmax,params)
        %create initial GP model
        if params.state_action_slicing_on==0
           
            BV = data;
            rew=reward;
            y= reward+params.gamma*Qmax;
            obs=y;
            current_size = size(data,2);
            K = kernel(data,data,sigma,params);
            Q = inv(K); 
            K = K + noise*eye(current_size);
            C = inv(K); 
            alpha = C*y';
            
        elseif params.state_action_slicing_on==1
            %if discrete state space just train separate Gaussian processes
            %for each DISCREET action
            %a=getaction(data,params);
            reset_vars(params);
            data=data(1:params.N_dim);
            for a=1:params.N_act
                [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a);
                
                BV(:,index) = data;
                rew(index)=reward;
                y= reward+params.gamma*Qmax;
                obs(index)=y;
                current_size(index) = size(data,2);
                
                noise_x = noise + 1;
                Q = y/noise_x;
                C = -1/noise_x;
                K = kernel(data,data,sigma,params);
                K = K + noise*eye(current_size);
                Q = inv(K);
                alpha = Q*y;
                [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
            end        
        end
    end

%------------------------------- predict -------------------------------
%  Given a new datapoint, predict a new value based on current model
    function [f,var_x] = predict(x)
       
        if params.state_action_slicing_on==0
            k = kernel(x,BV,sigma,params)';
            f = k'*alpha;
            var_x = kernel(x,x,sigma,params) + k'*C*k;
        elseif params.state_action_slicing_on==1
            a=getaction(x,params);
            x=x(1:params.N_dim);
            [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a);

            k = kernel(x,BV,sigma,params)';
            f = k'*alpha;
            kxx = kernel(x,x,sigma,params);
            var_x=kxx+ kxx'*C(1:current_size(index),1:current_size(index))*kxx;
       end
    end

%------------------------------- update --------------------------------
%  Given a new data pair, update the model; remember, this is passed
%  columnwise
    function update(x,reward,Qmax,params)
        % first compute simple upate quantities
        a=getaction(x,params);
        if params.state_action_slicing_on==1
            x=x(1:params.N_dim);
            [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a);
        end
        y = reward+params.gamma*Qmax;
        k_t1 = kernel(x,BV,sigma,params)';   % pg 9, eqs 30-31
        noise_x = noise + k_t1'*C*k_t1 + 1;
        q_t1 = (y - k_t1'*alpha)/(noise_x + noise);
        r_t1 = -1/(noise_x + noise);
        
        % compute residual projection update quantities
        e_t1 = Q*k_t1;
        gamma_t1 = double(1-k_t1'*e_t1);
        eta_t1 = 1/(1+gamma_t1*r_t1);
        
        if gamma_t1 < tol
            % in this case, addition of point to basis doesn't help much, so
            % don't add it, and compute update quantities in terms of old vectors
            % note that data, obs and gram matrix inverse not updated
            s_t1 = C*k_t1 + e_t1;
            alpha = alpha + q_t1*eta_t1*s_t1;
            C = C + r_t1*eta_t1*(s_t1*s_t1');
            if params.state_action_slicing_on==1 
                [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
            end

        else
            % in this case, you need to add the points
            current_size = current_size + 1;

            %in this case, you can simply add the points
            s_t1 = [C*k_t1; 1];
            alpha = [alpha; 0] + q_t1*s_t1;
            C = [C zeros(current_size-1,1); zeros(1,current_size)] + r_t1*(s_t1*s_t1');
            
            % update basis vectors and observations
            BV = [BV x];
            obs = [obs, y];
            rew = [rew, reward];
            
            % update Gram matrix and inverse
            K = [K k_t1; k_t1' 1];
            Q = inv(K);
            
            if current_size <= max_points
                %do nothing
            else
                if params.sparsification==1
                    scores = zeros(1,current_size);
                    for i=1:current_size
                        scores(i) = abs(alpha(i))/Q(i,i);
                    end
                    
                    %find index of minimum vector
                    [~, index] = min(scores);
                elseif params.sparsification==2
                    %just implement a windowed buffer
                    index=index+1;
                    if index==max_points
                        index=1;
                    end
                end
                
                %first compute scalar parameters
                a_s = alpha(index);
                c_s = C(index,index);
                q_s = Q(index,index);
                
                %compute vector parameters
                C_s = C(:,index);
                C_s(index) = [];
                Q_s = Q(:,index);
                Q_s(index) = [];
                
                %shrink matrices
                alpha(index) = [];
                C(:,index)   = [];
                C(index,:)   = [];
                Q(:,index)   = [];
                Q(index,:)   = [];
                K(:,index)   = [];
                K(index,:)   = [];
                
                %finally, compute updates
                alpha = alpha - (a_s/q_s)*(Q_s);
                C = C + (c_s/(q_s^2))*(Q_s*Q_s') - (1/q_s)*(Q_s*C_s' + C_s*Q_s');
                Q = Q - (1/q_s)*(Q_s*Q_s');
                
                current_size = current_size - 1;
                BV(:,index) = [];
                obs(index) = [];
                rew(index) = [];
            end
            
            %recalculate the estimate of Qstar
            if params.state_action_slicing_on==1
                [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store);
            end
        end
    end

%-------------------------------- get --------------------------------
%  Get a requested member variable.
    function mval = oGP_get(mfield)
        
        switch(mfield)
            case {'basis','BV'}
                mval = BV;
            case {'obs'}
                mval = obs;
            case {'K','kernel'}
                mval = K;
            case {'Q'}
                mval = Q;
            case {'current_size','size','current size'}
                mval = current_size;
            case {'alpha'}
                mval=alpha;
            case {'BV_store'}
                mval=BV_store;
            case {'alpha_store'}
                mval=alpha_store;
            case {'obs_store'}
                mval=obs_store;
        end
        
    end
    function [BV,K,alpha,C,Q,current_size,obs,rew,index]=assign_vars(a) 
        current_size=current_size_store(a);        
        BV=BV_store(:,max_points*(a-1)+1:max_points*(a-1)+current_size);
        K=K_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size);
        alpha=alpha_store(a,1:current_size)';
        C=C_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size);
        Q=Q_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size);
        
        obs=obs_store(a,1:current_size);
        rew=rew_store(a,1:current_size);
        index=index_store(a);
    end
    function [BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store]=assign_vars_back(a,BV_store,K_store,alpha_store,C_store,Q_store,current_size_store,obs_store,rew_store,index_store)
        current_size_store(a)=current_size;        
        BV_store(:,max_points*(a-1)+1:max_points*(a-1)+current_size)=BV;
        K_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size)=K;
        alpha_store(a,1:current_size)=alpha;
        C_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size)=C;
        Q_store(1:current_size,max_points*(a-1)+1:max_points*(a-1)+current_size)=Q;
        obs_store(a,1:current_size)=obs;
        rew_store(a,1:current_size)=rew;
        index_store(a)=index;
    end
    function reset_vars(params)
        BV_store           = zeros(params.N_dim,max_points*params.N_act);
        K_store            = zeros(max_points,max_points*params.N_act);
        alpha_store        = zeros(params.N_act,max_points);
        C_store            = K_store;
        Q_store            = K_store;
        current_size_store = ones(1,params.N_act);
        obs_store          = zeros(params.N_act,max_points);
        rew_store          = zeros(params.N_act,max_points);
        index_store        = ones(1,params.N_act);
    end
end

%============================ Helper Functions ===========================
%------------------------------- kernel ------------------------------
%y is a matrix of centers
function v =  kernel(xin,yin,sigma,params)

if params.state_action_slicing_on == 0
    x=xin;
    y=yin;
    if(length(sigma) == 1)
        d=x'*y;
        dx = sum(x.^2,1);
        dy = sum(y.^2,1);
        val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
        v = exp(-val./(2*sigma^2));
    else
        d =  (x'/diag(sigma.^2))*y;
        dx = sum((x'/diag(sigma.^2))'.*x,1);
        dy = sum((y'/diag(sigma.^2))'.*y,1);
        val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
        v = exp(-val./2);
    end
elseif params.state_action_slicing_on==1
    x=xin(1:params.N_dim);
    y=yin(1:params.N_dim,:);
    if(length(sigma) == 1)
        d=x'*y;
        dx = sum(x.^2,1);
        dy = sum(y.^2,1);
        val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
        vtemp = exp(-val./(2*sigma^2));
    else
        isigma = inv(diag(sigma.^2));
        d =  (x'*isigma)*y;
        dx = sum((x'*isigma)'.*x,1);
        dy = sum((y'*isigma)'.*y,1);
        val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
        vtemp = exp(-val./2);
    end
    v=vtemp;
end
end

%------------------------------- getaction ------------------------------
function a= getaction(xin,params)
  a=xin(params.N_dim+1:end);
end