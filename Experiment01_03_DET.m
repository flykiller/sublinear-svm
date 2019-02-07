% Run Mirror Prox for primal convex problem and save to file

clear all
clc
close all

rng(1);

d = 1000;      %number of features
k = d;      %number of classes
n = d;    %number of datapoints;
T = 3000;
gammasDet = zeros(1,T);     %stepsizes, calculate later;
GenerateData = 1;

Points_Plot = zeros(T,1);
plot_step_size = 0.1;   
p_toP = unique([round(10.^(0:plot_step_size:log10(T)))]);         %points where plot graph
num_ofP = length(p_toP);                             %number of points where to plot graph
for i = 1:num_ofP
   Points_Plot(p_toP(i)) = 1; 
end

lambda = 1/d;     
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

for t=1:T
    gammasDet(t) = 1/Lip;
end

PowerT = 6;
Repeat = 1;

SolDualGaps = zeros(num_ofP,Repeat);
SolPrimGaps = zeros(num_ofP,Repeat);
Times = zeros(num_ofP,Repeat);
Points = zeros(num_ofP,1);

Uhat = ones(2*d,k)/(2*d*k)*R1;      %initialization of matrix Uhat
V    = ones(n,k)/(k);               %initialization of matrix V
UhatAverage = Uhat;
VAverage = V;

sss = 1;
tic

for i = 1:T
    
    GradientV = Xhat*Uhat-Y;
    
    GradientU = (1/n*(V-Y)'*Xhat)';
    [UhatTemp,VTemp] = OneIterationDET_SVM(Uhat,V,d,n,k,gammasDet(i),...
        lambda,R1,GradientV,GradientU);
    
    GradientVNew = Xhat*UhatTemp - Y;
    
    GradientUNew = (1/n*(VTemp-Y)'*Xhat)';
    [Uhat,V] = OneIterationDET_SVM(Uhat,V,d,n,k,gammasDet(i),lambda,...
        R1,GradientVNew,GradientUNew);
    
    UhatAverage = UhatAverage*(i)/(i+1)+Uhat/(i+1);
    VAverage = VAverage*(i)/(i+1)+V/(i+1);
    if Points_Plot(i) == 1
        E_T = toc;
        if sss == 1
            Times(sss) = E_T;
            display(Times(sss));
        else
            Times(sss) = Times(sss-1) + E_T;
            display(Times(sss));
            
        end
        [tmp1,tmp2] = Evaluate_Duality_Gap(Xhat,Y,UhatAverage,VAverage,n,lambda, R1);
        SolPrimGaps(sss) = tmp1;
        SolDualGaps(sss) = tmp2;
        sss = sss + 1;
        tic
    end
    if mod(i,10) == 0
        save('Exp_01_Det_n_1000_T_3000_lambda_1_1000t',...
            'SolDualGaps', 'SolPrimGaps', 'Times');
        display(i);
    end
end

toc