% Run fully stochastic SVM on synthetic data and save to file

clear all
clc
close all
rng(1);

d = 1000;      %number of features
k = 1000;      %number of classes
n = 1000;      %number of datapoints;

PowerT = 6.5;    %maximum power of T to do simulations
Repeat = 10 ;    %number of replicates to average
GenerateData = 1;
lambda = 0.001;                 %regularization coefficient;
UStar = eye(k)/2;               %initialize optimal U
OmegaV = n;                         %Radius of set V;
R1 = sum(sum(abs(UStar)));    
OmegaU = R1^2;                      %Radius of set U;


if GenerateData == 1
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
    
    Xhat=[X,-X];
    
    clear X;
    Info_name = 'INFO_' + string(n);
    %save(Info_name, 'Xhat', 'Y', 'Ytilde', 'L_U_V', 'Lip');
else
    Info_name = 'INFO_' + string(n);
    load Info_name;
end

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

SolDualGaps = zeros(2*PowerT-1,Repeat);
SolPrimGaps = zeros(2*PowerT-1,Repeat);
Times = zeros(2*PowerT-1,Repeat);
Points = zeros(2*PowerT-1,1);

T = 10^PowerT;

for iter = 1:Repeat
    for i = 1:0.5:PowerT
        T = floor(10^i);
        display(T);
        Points(2*i-1) = T;
        gammasStoc = ones(1,T)*1/sqrt(T)*1/(Lip+sqrt(Theta))*2;
        Points_Plot = zeros(T,1);
        Points_Plot(T) = 1;
        [a,b,c,E_T] = Function_Full_SVM(n,d,k,T,lambda,R1,gammasStoc,Points_Plot,1, Xhat,...
            Y,Ytilde, TauV, TauU,1);
        SolPrimGaps(i*2-1,iter) = a;
        SolDualGaps(i*2-1,iter) = b;
        Times(i*2-1,iter) = E_T;
        save('Exp_01_FULL_n_1000_T_3000000_lambda_1_1000',...
            'Points', 'SolDualGaps', 'SolPrimGaps', 'Times');
    end
end
