%test if we have linear dependence.

clear all
clc
close all

rng(1);
NUM_ITER = 3;   %number of iterations to average
NUM_POWER = 2;  %number of powers to evaluate: in our case 400, 800, 1600, 3200 and 6400
                %note, that 6400 can be used only with >=8Gb RAM, because
                %memory swap effects reduce the efficiency.
                
times = zeros(NUM_ITER,NUM_POWER);
Sizes = zeros(1,NUM_POWER);
                
step = 2;       %step to increase
T = 10000;      %number of iterations
d = 400; k = d; n = d;
for ii = 1:NUM_POWER
    for iter = 1:NUM_ITER

        lambda = 0.00;    
        UStar = eye(k)/2;               %initialize optimal U
        OmegaV = n;                         %Radius of set V;
        R1 = sum(sum(abs(UStar)))*2;    
        OmegaU = R1^2;                      %Radius of set U;
        X      = randn(n,d);         % generate data X as standard normal
        Y      = zeros(n,k);         % responses
        Ytilde = zeros(n,1);         % responses in sparse view.

        for i = 1:n
           classes = UStar'*X(i,:)' + 1/sqrt(k)*randn(k,1);
           [a,num] = max(classes);
           Y(i,num) = 1;
           Ytilde(i) = num;
        end

        L_U_V = 0;
        for i=1:d
            L_U_V = max(L_U_V,norm(X(:,i),'fro'));
        end
        L_U_V = L_U_V/n;                     %Cross Lipshitz constant;
        Lip = L_U_V*sqrt(OmegaU*OmegaV);     %Full Lipshitz constant;

        Xhat = [X,-X];

        TauV = zeros(1,2*d);                %precalculated norms for U
        TauU = zeros(1,n);                  %precalculated norms for V

        for i = 1:2*d
           TauV(i) = norm(Xhat(:,i),'fro'); 
        end

        for i = 1:n
            TauU(i) = norm(Xhat(i,:), Inf);
        end

        sigmaU = 4*L_U_V^2*R1^2;
        sigmaV = 2*n*L_U_V^2 + 2*sum(max(Xhat'));

        Theta = 2*(OmegaV*sigmaU+OmegaU*sigmaV)/n;

        %--------------------------------------------------------%
        %     Stepsizes                                          %
        %--------------------------------------------------------%

        gammasStoc = ones(1,T)*1/sqrt(T)*1/(Lip+sqrt(Theta));
        Points_Plot = zeros(T,1);

        [a,b,c,dd] = Function_Full_SVM(n,d,k,T,lambda,R1,gammasStoc,Points_Plot,1, Xhat,...
            Y,Ytilde, TauV, TauU, 0);

        times(iter,ii) = dd;
        
        display(d);    
    end
    
    Sizes(ii) = n;
    n = n*step;
    d = d*step;
    k = k*step;
end

save Table01 times Sizes;