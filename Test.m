clear all
clc
close all

rng(1)
d = 2048;
n = 2000;
k = 1000;

%load imagenet_features_processed.mat

 X      = randn(n,d); 

L_U_V = 0;
for i=1:d
    L_U_V = max(L_U_V,norm(X(:,i),'fro'));
end

s1 = L_U_V/sqrt(n);

s2 = sum(max(X'))/n;