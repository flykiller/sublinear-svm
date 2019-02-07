% Run SGD for primal convex problem and save to file

clear all
clc
close all

rng(1);

n = 1000;      %number of features
k = 1000;      %number of classes
d = 1000;    %number of datapoints;
GenerateData = 1;
lambda = 1/n;     %regularization coefficient;
UStar = eye(k)/2;               %initialize optimal U
OmegaV = n;                         %Radius of set V;
R1 = sum(sum(abs(UStar)));    
OmegaU = R1^2;                      %Radius of set U;

if GenerateData == 1
    X      = randn(n,d);         % generate data X as standard normal
    Y      = zeros(n,k);         % responses
    Ytilde = zeros(n,1);         % responses in sparse view.

    for i = 1:n
       classes = UStar'*X(i,:)' + randn(k,1)/sqrt(k);
       [a,num] = max(classes);
       Y(i,num) = 1;
       Ytilde(i) = num;
    end
  
    Info_name = 'INFO_pure' + string(n);
    %save(Info_name, 'X', 'Y', 'Ytilde');
else
    Info_name = 'INFO_pure' + string(n);
    load Info_name;
end

Repeat = 1;
PowerT = 5;
T = 10^PowerT;

name_to_save = 'SGD_n_' + string(n) + '_T_' + string(T) +...
    '_lambda_1_' + string(1/lambda);

SolPrimGaps = zeros(2*PowerT-1,Repeat);
Times = zeros(2*PowerT-1,Repeat);
Points = zeros(2*PowerT-1,1);

for iter = 1:Repeat
    for i = 1:0.5:PowerT
        T = floor(10^i);
        display(T);
        Points(2*i-1) = T;
        
        step =  norm(UStar,'fro') * sqrt(n)/sqrt(T)/norm(X,'fro');
        gammasStoc = ones(T,1)*step;
        [Time,Uout, UAverageOut] = Function_Primal_SGD(X,Y,Ytilde,d,n,k,...
            gammasStoc,lambda,T);
        SolPrimGaps(i*2-1,iter) = Evaluate_Primal_Gap(X,Y,Uout,n,lambda);
        Times(i*2-1,iter) = Time;
        save('Exp_01_SGD_n_1000_T_100000_lambda_1_1000',...
            'Points', 'SolPrimGaps', 'Times');
    end        
end