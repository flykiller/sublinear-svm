function [Sum1, Sum2] = Evaluate_Duality_Gap(Xhat,Y,Uhat_t,V_t,n,lambda, R1)
%% function to evaluate duality gap for multi-class SVM.
%Xhat - data, [X,-X]
%Y - the matrix of responses
%(Uhat_t, V_t) - the point where evaluate gap
%n - sample size
%lambda - regularization parameter
%R1 - l1 radius of optimal solution

%%Sum1 is primal accuracy
% Sum2 is dual accuracy

    S = (Xhat*Uhat_t - Y)/n;
    Sum1 = 1;       
    for i = 1:n
        Sum1 = Sum1 + max(S(i,:));
    end
    Sum1 = Sum1  - 1/n*trace(Y'*Xhat*Uhat_t) + lambda*sum(sum(Uhat_t));

    Sum2 = 1;
    Sum2 = Sum2 - 1/n*trace(V_t'*Y) ;
    T = -Xhat'*(V_t-Y)/n - lambda;
    Sum2 = Sum2 - max(0,max(max(T))*R1);
    Out = Sum1 - Sum2;
    display([Out,Sum1,Sum2]);
end

