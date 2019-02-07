function [PrimalGap] = Evaluate_Primal_Gap(X,Y,U,n,lambda)
%%function to evaluate Primal accuracy for multiclass SVM problem
%X,Y - data and responces
%U - current point to evaluate
%n - sample size
%lambda - regularization parameter

%%PrimalGap - primal accuracy

    S = (X*U - Y)/n;
    Sum1 = 1;
    for i = 1:n
        Sum1 = Sum1 + max(S(i,:));
    end
    PrimalGap = Sum1  - 1/n*trace(Y'*X*U) + lambda*sum(sum(abs(U)));
    display(PrimalGap);

end

