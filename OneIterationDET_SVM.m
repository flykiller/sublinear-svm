function [UhatNew,VNew] = OneIterationDET_SVM(Uhat,V,d,n,k,gamma,lambda,R1,GradientV,GradientU)
%% one iteration for determenistic SVM problem
% Uhat, V - current point
% d, n, k - dimensions
% gamma - stepsize
% lambda - regularization coefficient
% R1 - l1 radius of optimal solution
% GradientV, GradientU - gradients, where evaluate

%% Uhat, V - new values of point

    s1 = log(k);
    s2 = log(2*k*d);
    
    Q = GradientV;
    Qnew = zeros(n,k);
    for i=1:n
        for j=1:k
            Qnew(i,j) = V(i,j)*exp(2*gamma*s1*Q(i,j));
        end
    end
    
    VNew = zeros(n,k);
    UhatNew = zeros(2*d,k);

    QnewSum = sum(Qnew,2);
    for i=1:n
        for j=1:k
            VNew(i,j) = Qnew(i,j)/QnewSum(i);
        end
    end
    
    S = GradientU;

    M = 0;
    for i=1:2*d
        for j=1:k
            M = M + Uhat(i,j)*exp(-2*gamma*s2*R1*S(i,j));
        end
    end
    renorm = min(M*exp(-2*lambda*s2*gamma*R1),R1);
    for i=1:2*d
        for j = 1:k
            UhatNew(i,j) = Uhat(i,j)*exp(-2*gamma*s2*R1*S(i,j))/M*renorm;
        end
    end
end


