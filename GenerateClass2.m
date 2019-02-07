function [ Num ] = GenerateClass2(K, classes )
%Function to sample from any discrete distribution with given probabiliteis
%classes and number of outcomes K
    TT = rand;
    Num = K + 1 - sum((cumsum(classes) >= TT));
    if Num > K
        Num = K;
    end
end

