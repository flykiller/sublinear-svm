function [SolGapPrimal,SolGapDual, FixData, Time_Spent] = Function_Full_SVM(n,d,k,T,...
    lambda, R1, gammasStoc, Points_Plot, NPoints_Plot, Xhat, Y, Ytilde, SIGMA,...
    TAU, EvaluateGaps)
    
    %% INPUT PARAMETERS
    % n - sample size
    % d - number of features
    % k - number of classes
    % T - number of iterations
    % lambda - regularization parameter
    % R1 - radius of optimal solution
    % gammasStoc - stepsizes for all T iterations
    % Point_Plot - points where evaluate Duality Gap
    % NPoints_Plot - number of non-zero elements in Point_Plot
    % Xhat - data matrix = [X, -X]
    % Y - response matrix
    % Ytilde - numbers of classes
    % SIGMA - Frobenius norms of rows of matrix Xhat
    % TAU - infinity norms of column of matrix Xhat
    % EvaluateGaps - is duality gaps will be evaluated, debug option
    
    %% OUTPUT PARAMETERS
    %SolGapPrimal - values of Primal Gaps
    %SolGapDual - values of Dual Gaps
    %FixData - chech how often we need to rescale data to avoid loss of
    %accuracy
    %Time_Spent - times spent for the main part of algorithm.
    
    %% Preprosessing
    SolGapPrimal = zeros(NPoints_Plot,1);
    SolGapDual = zeros(NPoints_Plot,1);
    
    UhatTilde = ones(2*d,k)/(2*d*k)*R1;      %initialization of matrix Uhat
    VTilde    = ones(n,k)/(k);               %initialization of matrix V
    rho = ones(1,2*d)*R1/(2*d);              
    alpha = ones(1,2*d);
    beta = ones(1,n);
    alphaSum = zeros(1,2*d);
    betaSum = zeros(1,n);
    alphaStarts = zeros(2*d,k);
    USum = zeros(2*d,k);
    betaStarts = zeros(n,k);
    VSum = zeros(n,k);
    FixData = zeros(T,1);
    
    VTilde_L_M = zeros(n,1);
    VSum_L_M = zeros(n,1);
    betaStarts_L_M = zeros(n,1);
    for i = 1:n
        VTilde_L_M(i) =  VTilde(i,Ytilde(i));
        VSum_L_M(i)   =  VSum(i,Ytilde(i));
        betaStarts_L_M(i)   =  betaStarts(i,Ytilde(i));
    end
    sss = 1;
    
    tic
    
    for t = 1:T
        %% ------------------------------------------------------%
        %     Algorithm generate samples for U                   %
        %--------------------------------------------------------%
        
        Sumprobs = sum(TAU);
        ProbsPV = TAU/Sumprobs;
        iU = GenerateClass2(n,ProbsPV);   
        VTilde(iU,Ytilde(iU)) = VTilde_L_M(iU);     %make sure to update
        ProbsPV2 = VTilde(iU,:)*beta(iU);
        jU = GenerateClass2(k,ProbsPV2);   
        SU1 = Xhat(iU,:)'/ ProbsPV(iU)/n;
        SU2 = -SU1;
        
        
        %% -------------------------------------------------------%
        %     Algorithm generate samples for V                   %
        %--------------------------------------------------------%
         
        Sumprobs = SIGMA*rho';
        ProbsPU = SIGMA.*rho/Sumprobs;
        iV = GenerateClass2(2*d,ProbsPU);   
        ProbsPU2 = UhatTilde(iV,:)*alpha(iV)/rho(iV);
        jV = GenerateClass2(k,ProbsPU2);
        SV = Xhat(:,iV) * Sumprobs/SIGMA(iV);
        
        %% ------------------------------------------------------%
        %    Algorighm Evaluate nu                               %
        %--------------------------------------------------------%
        YtildeiU = Ytilde(iU);
        if (YtildeiU == jU)
            M = sum(rho);
        else
            q1 = exp(SU1.*(-2*gammasStoc(t)*log(2*k*d)*R1));
            q2 = exp(SU2.*(-2*gammasStoc(t)*log(2*k*d)*R1));
            M = sum(rho) - sum(alpha'.*UhatTilde(:,jU).*(1-q1)) - ...
                sum(alpha'.*UhatTilde(:,YtildeiU).*(1-q2)); 
        end
        
        nu = min(M*exp(-2*lambda*log(2*d*k)*gammasStoc(t)*R1),R1)/M;
        
        %% ------------------------------------------------------%
        %    Algorithm Udpates for U                             %
        %--------------------------------------------------------%
        
        USum(:,jU)=USum(:,jU)+UhatTilde(:,jU).*(alphaSum' + alpha'-alphaStarts(:,jU));
        if (YtildeiU ~= jU)
           USum(:,YtildeiU) = USum(:,YtildeiU) + UhatTilde(:,YtildeiU).*...
               (alphaSum' + alpha' - alphaStarts(:,YtildeiU));
        end
        
        alphaStarts(:,jU) = alphaSum + alpha;
        alphaStarts(:,YtildeiU) = alphaSum + alpha;
        alphaSum = alphaSum + alpha;
        
        if (YtildeiU == jU)
            rho = rho*nu;
        end
           
        if (YtildeiU ~= jU)
            rho = nu*(rho + UhatTilde(:,jU)'.*alpha.*(q1-1)' + ...
                UhatTilde(:,YtildeiU)'.*alpha.*(q2-1)');
            UhatTilde(:,jU) = UhatTilde(:,jU).*q1(:);
            UhatTilde(:,YtildeiU) = UhatTilde(:,YtildeiU).*q2(:); 
        end
           
        
        alpha = alpha*nu;
        
        %--------------------------------------------------------%
        %    Algorithm Udpates for V                             %
        %--------------------------------------------------------%
        
        sigmaTilde = exp(SV.*(2*gammasStoc(t)*log(k)));
        VSum_L_M(:)=VSum_L_M(:)+VTilde_L_M(:).*(betaSum(:)+beta(:)-betaStarts_L_M(:));        
        tt1 = (Ytilde~=jV);
        VSum(:,jV)=VSum(:,jV)+VTilde(:,jV).*(betaSum(:)+beta(:)-betaStarts(:,jV)).*tt1(:); 
        betaStarts_L_M(:) = betaSum(:)+beta(:);
        betaStarts(:,jV) = betaSum + beta;
        betaSum = betaSum + beta;
        
        LL = (1 - exp(-2*gammasStoc(t)*log(k)));
        xi2 = (1 - beta(:).*VTilde(:,jV).*(1 - sigmaTilde(:)) - beta(:).*VTilde_L_M(:)*LL).*tt1(:);
        xi3 = (1-beta(:).*VTilde_L_M(:).*(1-sigmaTilde(:)*(1-LL))).*(1-tt1(:));
        xi = xi2+xi3;
        
        VTilde_L_M(:) = VTilde_L_M(:).*(sigmaTilde(:).*exp(-2*gammasStoc(t)*log(k))).^(1-tt1(:));
        VTilde_L_M(:) = VTilde_L_M(:).*(exp(-2*gammasStoc(t)*log(k)).^tt1(:));
        VTilde(:,jV) = VTilde(:,jV).*sigmaTilde(:).*tt1(:);
        beta = beta./xi';

        %--------------------------------------------------------%
        %    Check if data is exploded. CHECK!!!!                %
        %--------------------------------------------------------%
        
        if (min(alpha) < 1e-4)||(max(alpha) > 1e4)||(min(beta) < 1e-4)||(max(beta) > 1e4)
            FixData(t) = 1;
            USum = USum + UhatTilde .* (repmat(((alphaSum + alpha)'),1,k) - alphaStarts);
            
            UhatTilde = UhatTilde.*repmat(alpha',1,k);
            alphaStarts = zeros(2*d,k);
            alpha = ones(1,2*d);
            alphaSum = zeros(1,2*d);
            
            for i = 1:n
                VTilde(i,Ytilde(i)) = VTilde_L_M(i);
                VSum(i,Ytilde(i)) = VSum_L_M(i);
                betaStarts(i,Ytilde(i)) = betaStarts_L_M(i);
            end
            
            VSum = VSum + VTilde.* (repmat(((betaSum + beta)'),1,k) - betaStarts);
            betaStarts = zeros(n,k);
            VtildeSum = sum(VTilde,2)';
            beta = ones(1,n);
            betaSum = zeros(1,n);
            VTilde = VTilde./repmat(VtildeSum',1,k);
            
            for i = 1:n
                VTilde_L_M(i) =  VTilde(i,Ytilde(i));
                VSum_L_M(i) =  VSum(i,Ytilde(i));
                betaStarts_L_M(i) = betaStarts(i,Ytilde(i));
            end
            
        end
        
        %--------------------------------------------------------%
        %    Evaluate duality gaps                               %
        %--------------------------------------------------------%
        
        if EvaluateGaps == 1
            if Points_Plot(t) == 1
                
                for i = 1:n
                    VTilde(i,Ytilde(i)) = VTilde_L_M(i);
                    VSum(i,Ytilde(i)) = VSum_L_M(i);
                    betaStarts(i,Ytilde(i)) = betaStarts_L_M(i);
                end
                
                UFinalized = (USum + UhatTilde .* ...
                    (repmat(((alphaSum + alpha)'),1,k) - alphaStarts))/(t+1);

                VFinalized = (VSum + VTilde .* ...
                    (repmat(((betaSum + beta)'),1,k) - betaStarts))/(t+1);

                [tmp1,tmp2] = Evaluate_Duality_Gap...
                    (Xhat,Y,UFinalized,VFinalized,n,lambda, R1);
                SolGapPrimal(sss) = tmp1;
                SolGapDual(sss) = tmp2;
                sss = sss + 1;
            end
        end
        
    end
    Time_Spent = toc;
end