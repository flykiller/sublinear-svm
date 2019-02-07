function [Time, U, UAverage ] = Function_Primal_SGD(X,Y,Ytilde,d,n,k,gammas,lambda,T)
%% Primal SGD for multiclass SVM probmel
%X,Y,Ytilde - data and responces
%d,n,k - dimension, sample size and number of classes
%gammas - stepsizes
%lambda - regularization coefficient
%T - number of iterations

%%Time - time spent by algorithm
%U - output value of U
%UAverage - averaged value of U

    tic
    U = zeros(d,k);         %initialization of first iteration, start from zero point
    UAverage = U;
    
    for iter = 1:T
        i = randsample(n,1);
        yi = Y(i,:);
        xi = X(i,:);
        ss = U'*xi' + 1;
        ss(Ytilde(i)) = ss(Ytilde(i)) - 1;
        [~,b] = max(ss);
        deltaYk = zeros(1,k);
        deltaYk(b) = 1;

        U = U - gammas(iter)*xi'*(deltaYk-yi); 
        U = sign(U).*(abs(U)-gammas(iter)*lambda);
        UAverage = UAverage*(iter)/(iter+1) + U/(iter+1);
    end
    Time = toc;

end