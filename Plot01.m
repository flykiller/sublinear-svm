%plot 01 - construct the 1st plot from the article: non-logariphmic scale

clear all
clc
close all

lw=4; %Linewidth
fs=30; %Fontsize
fw='Normal'; %FontWeight
fsa=23; %Fontsize
marksize = 13;

Fstar = 0.3374; %estimation of Fstar, obtained by 10^8 iterations run.

load Exp_01_FULL_n_1000_T_3000000_lambda_1_1000;
startP = 3;
finishP = length(SolPrimGaps);
hold on

%Fully stochastic, duality gap
plot(mean(Times(startP:finishP,:),2),mean(SolPrimGaps(startP:finishP,:),2)-...
mean(SolDualGaps(startP:finishP,:),2),':o','MarkerSize',marksize,...
'MarkerEdgeColor','b','MarkerFaceColor','b','Color','blue','Linewidth',lw);

%Fully stochastic, primal gap
 plot(mean(Times(startP:finishP,:),2),SolPrimGaps(startP:finishP)...
   -Fstar,'-o','MarkerSize',marksize,'MarkerEdgeColor','b',...
'MarkerFaceColor','b','Color','blue','Linewidth',lw);

load('Exp_01_SGD_n_1000_T_100000_lambda_1_1000');
startP = 3;
finishP = length(SolPrimGaps);

%SGD for primal problem
plot(mean(Times(startP:finishP,:),2),mean(SolPrimGaps(startP:finishP,:),2)-Fstar,...
    '-d','MarkerSize',marksize,'MarkerEdgeColor','black',...
'MarkerFaceColor','black','Color','black','Linewidth',lw);  

load('Exp_01_Det_n_1000_T_3000_lambda_1_1000');
startP = 1;
finishP = length(SolPrimGaps);

%Determenistic Mirror Prox, duality gap
plot(mean(Times(startP:finishP,:),2),mean(SolPrimGaps(startP:finishP,:),2)-...
mean(SolDualGaps(startP:finishP,:),2),':','Color','red','Linewidth',lw);

%Determenistic Mirror Prox, primal value
plot(mean(Times(startP:finishP,:),2),mean(SolPrimGaps(startP:finishP,:),2)-Fstar...
   ,'-','Color','red','Linewidth',lw); 

%axis([6,1000,0.03,14])
axis([0,500,0.06,6])
set(gca,'FontWeight',fw,'FontSize',fsa);
legend('Full-SMD, Gap','Full-SMD, Acc.','SSM, Acc.',...
'MP, Gap', 'MP, Acc.');
l2 = xlabel('Runtime (sec)','FontSize',fs,'FontWeight',fw);
l= ylabel('Error','FontSize',fs,'FontWeight',fw);
set(l,'Interpreter','Latex');
set(l2,'Interpreter','Latex');


