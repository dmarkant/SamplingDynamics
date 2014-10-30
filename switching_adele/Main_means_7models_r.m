%                    Quantiles,  probabilities  2 Attribute
clear;
clc

% As a function of r (w) 
r=0.002:0.001:0.02;
nw=length(r);
%p=[ -.1 .1 .0]; 
p=[ -.1 ]; 
pt=length(p);
tau = 1;
for kk = 1:nw   %
     

for k= 1:pt  %something to vary in the para vector 
   


para=[   p(k)  .05 .00 r(kk) r(kk) 10];

%para=[  p(k)  .05 .00 r(kk) .02-r(kk) 10];


%--------------------------------------------------------------------------
%
%---------------------------------------------
 

    
   [proba1, probb1, eta1, etb1] = c_means_m1(para,tau);
      [proba2, probb2, eta2, etb2]= c_means_m2(para,tau);
       [proba3, probb3, eta3, etb3]= c_means_m3(para,tau);
        [proba4, probb4, eta4, etb4] = c_means_m4(para,tau);
%         [proba5, probb5, eta5, etb5]= c_means_m5(para,tau);
%          [proba6, probb6, eta6, etb6] = c_means_m6(para,tau);
           %[proba7, probb7, eta7, etb7]= c_means_m7(para,tau);
           
           
pA1(kk,k) =proba1;
pB1(kk,k) =probb1;
etA1(kk,k)= eta1;
etB1(kk,k)= etb1; 
pA2(kk,k) =proba2;
pB2(kk,k) =probb2;
etA2(kk,k)= eta2;
etB2(kk,k)= etb2; 
pA3(kk,k) =proba3;
pB3(kk,k) =probb3;
etA3(kk,k)= eta3;
etB3(kk,k)= etb3; 
pA4(kk,k) =proba4;
pB4(kk,k) =probb4;
etA4(kk,k)= eta4;
etB4(kk,k)= etb4; 
% pA5(kk,k) =proba5;
% pB5(kk,k) =probb5;
% etA5(kk,k)= eta5;
% etB5(kk,k)= etb5; 
% pA6(kk,k) =proba6;
% pB6(kk,k) =probb6;
% etA6(kk,k)= eta6;
% etB6(kk,k)= etb6; 
% pA7(kk,k) =proba7;
% pB7(kk,k) =probb7;
% etA7(kk,k)= eta7;
% etB7(kk,k)= etb7; 
end
end   
      
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
fs=12;
lw=2;
fn='Arial';
x=0:50:nw;
ms=3;
xt=[1 11 16 19];
xL=[ 50 100 200 500];
yt=[10 20 30 40 50 60 70 80 90 100 110 120 130 140];
yL=yt.*tau;
 max=120;  
ep=3;
%figure(1);
subplot(2,2,1)

    plot(pA1, 'r-', 'linewidth',lw); 
    hold on;
   plot(pA2, 'b-', 'linewidth',lw); 
     hold on;
     plot(pA3, 'g-', 'linewidth',lw);  
     hold on;
     plot(pA4, 'm-', 'linewidth',lw);  
%     hold on;
%     plot(pA5, 'c-', 'linewidth',lw);  
%     hold on;
%     plot(pA6, 'k-', 'linewidth',lw);  
%     hold on;
%     plot(pA7, 'y-', 'linewidth',lw);
%     hold on;
   %plot(pB1, 'r--', 'linewidth',lw); 
%     hold on;
%     plot(pB2, 'b--', 'linewidth',lw); 
%     hold on;
%     plot(pB3, 'g--', 'linewidth',lw);  
%     hold on;
%     plot(pB4, 'm--', 'linewidth',lw);  
%     hold on;
%     plot(pB5, 'c--', 'linewidth',lw);  
%     hold on;
%     plot(pB6, 'k--', 'linewidth',lw);  
%     hold on;
%     plot(pB7, 'y--', 'linewidth',lw);
    axis([0 nw 0 1]);
    set(gca,'XTick', xt);
    set(gca,'XTickLabel',xL);
    xlabel('1/r_{1}','fontsize', fs,'FontName', fn);%, 'fontName','Times New Roman');
    ylabel('Choice prob A ', 'FontSize', fs); 
%     text(-3, 1.05, '0.3','FontSize',fs,'FontName', fn);
%     text(94, 1.05, '0.2','FontSize',fs,'FontName', fn);
%     text(194, 1.05, '0.1','FontSize',fs,'FontName', fn);
%     text(297, 1.05, '0','FontSize',fs,'FontName', fn);
%     text(147, 1.06, 'w_{21}','FontSize',fs,'FontName', fn);
    legend(gca,'one switch', 'one switch', 'back and forth', ' back and forth','orientation', 'horizontal'); 
    %legend(gca,'one', 'once', 'M3', 'M4', 'M5', 'M6', 'M7','orientation', 'horizontal','location', 'bestOutside' ); 
    %legend('boxoff');
    set(gca,'FontSize',fs,'FontName', fn);
    hold off;
   

%figure(2);
subplot(2,2,2)
    plot(etA1, 'r-', 'linewidth',lw); 
    hold on;
    plot(etA2, 'b-', 'linewidth',lw); 
    hold on;
    plot(etA3, 'g-', 'linewidth',lw);  
    hold on;
    plot(etA4, 'm-', 'linewidth',lw);  
%     hold on;
%     plot(etA5, 'c-', 'linewidth',lw);  
%     hold on;
%     plot(etA6, 'k-', 'linewidth',lw);  
%     hold on;
%     plot(etA7, 'y-', 'linewidth',lw);
    axis([0 nw 60 120]);
    %legend(gca,'a_1, w_{12}', 'a_2, w_{21}', 'a_1, w_{12} = w_{21}', 'a_2, w_{21} = w_{12}', '0.5 a_1, w_{12}', '0.5 a_1, w_{21}', '0.5 a_1, w_{12}=w_{21}','orientation','vertical'); 
    %legend(gca,'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7','orientation','vertical', -1); 
    set(gca,'XTick', xt);
    set(gca,'XTickLabel',xL);
%     text(-3, max+ep, '0.3','FontSize',fs,'FontName', fn);
%     text(94, max+ep, '0.2','FontSize',fs,'FontName', fn);
%     text(194, max+ep, '0.1','FontSize',fs,'FontName', fn);
%     text(297, max+ep, '0','FontSize',fs,'FontName', fn);
%     text(147, max+ep+1, 'w_{21}','FontSize',fs,'FontName', fn);
    xlabel('1/r_{1}','fontsize', fs,'FontName', fn);%, 'fontName','Times New Roman');
    ylabel('Mean RT for A ','FontSize',fs, 'FontName', fn);
    set(gca,'FontSize',fs,'FontName', fn);
    hold off; 
   % print -r300 -deps2c etA_102_w03
    %print -r300 -deps2c etA_021_w03
    %print -r300 -deps2c etA_1-05_w03
    
subplot(2,2,3)

   plot(pB1, 'r--', 'linewidth',lw); 
    hold on;
    plot(pB2, 'b--', 'linewidth',lw); 
    hold on;
    plot(pB3, 'g--', 'linewidth',lw);  
    hold on;
    plot(pB4, 'm--', 'linewidth',lw);  
%     hold on;
%     plot(pB5, 'c--', 'linewidth',lw);  
%     hold on;
%     plot(pB6, 'k--', 'linewidth',lw);  
%     hold on;
%     plot(pB7, 'y--', 'linewidth',lw);
    axis([0 nw 0 1]);
    set(gca,'XTick', xt);
    set(gca,'XTickLabel',xL);
    xlabel('1/r_{1}','fontsize', fs,'FontName', fn);%, 'fontName','Times New Roman');
    ylabel('Choice prob B ', 'FontSize', fs); 
%     text(-3, 1.05, '0.3','FontSize',fs,'FontName', fn);
%     text(94, 1.05, '0.2','FontSize',fs,'FontName', fn);
%     text(194, 1.05, '0.1','FontSize',fs,'FontName', fn);
%     text(297, 1.05, '0','FontSize',fs,'FontName', fn);
%     text(147, 1.06, 'w_{21}','FontSize',fs,'FontName', fn);
     %legend(gca,'a_1, w_{12}', 'a_2, w_{21}', 'a_1, w_{12} = w_{21}', 'a_2, w_{21} = w_{12}', '0.5 a_1, w_{12}', '0.5 a_1, w_{21}', '0.5 a_1, w_{12}=w_{21}','orientation','vertical' ,'location', 'bestOutside'); 
    %legend(gca,'one', 'once', 'M3', 'M4', 'M5', 'M6', 'M7','orientation', 'horizontal','location', 'bestOutside' ); 
    %legend('boxoff');
    set(gca,'FontSize',fs,'FontName', fn);
    hold off;
   


%figure(3);
subplot(2,2,4)
conditions = {'Different', 'Same', 'Neutral'};
    plot(etB1, 'r--', 'linewidth',lw); 
    hold on;
    plot(etB2, 'b--', 'linewidth',lw); 
    hold on;
    plot(etB3, 'g--', 'linewidth',lw);  
    hold on;
    plot(etB4, 'm--', 'linewidth',lw);  
 %   hold on;
%     plot(etB5, 'c--', 'linewidth',lw);  
%     hold on;
%     plot(etB6, 'k--', 'linewidth',lw);  
%     hold on;
%     plot(etB7, 'y--', 'linewidth',lw);
    axis([0 nw 60 120]);
    %legend(gca,'a_1, w_{12}', 'a_2, w_{21}', 'a_1, w_{12} = w_{21}', 'a_2, w_{21} = w_{12}', '0.5 a_1, w_{12}', '0.5 a_1, w_{21}', '0.5 a_1, w_{12}=w_{21}','orientation','vertical'); 
    set(gca,'XTick', xt);
    set(gca,'XTickLabel',xL);
    xlabel('1/r_{1}','fontsize', fs,'FontName', fn);%, 'fontName','Times New Roman');
    ylabel('Mean RT for B ','FontSize',fs, 'FontName', fn);
%     text(-3, max+ep, '0.3','FontSize',fs,'FontName', fn);
%     text(94, max+ep, '0.2','FontSize',fs,'FontName', fn);
%     text(194, max+ep, '0.1','FontSize',fs,'FontName', fn);
%     text(297, max+ep, '0','FontSize',fs,'FontName', fn);
%     text(147, max+ep+1, 'w_{21}','FontSize',fs,'FontName', fn);
    set(gca,'FontSize',fs,'FontName', fn);
    hold off; 
    %print -r300 -deps2c etB_102_w03
    %print -r300 -deps2c etB_021_w03
    %print -r300 -deps2c etB_1-05_w03
% 
% 
% %--------------------------------------------------------------------
% 
% 
% % 
% % 
% % 
% % 
% % %A is up, note, different from fixed_t
% % 
% figure(3)
% plot(pA(:,1:4), 'Linewidth', lw);
% hold on;
% plot(pA(:,5), 'k-', 'Linewidth', lw);
% hold on;
% %plot(x,pBs(1,:)+0*x, 'k-*', 'Linewidth', lw);
% %hold on;
% plot(pB(:,1:4),'--','Linewidth', lw);
% hold on;
% %plot(pB(:,5), 'k--', 'Linewidth', lw)
% hold on;
% %plot(x,pAs(1,:)+0*x, 'k--*', 'Linewidth', lw);
% hold off;
% axis([0 nw 0 1]);
% xlabel('Attention switching  w_{12}','FontSize',fs, 'FontName', fn);
% set(gca, 'XTick', xt, 'FontSize', fs, 'FontName', fn);
% set(gca, 'XTickLabel', xL,'FontSize', fs,'FontName', fn);
% ylabel('Choice probability','FontSize',fs, 'FontName', fn);
% %legend('\mu_{2} = 0.02', '\mu_{2} = 0.04', '\mu_{2} = 0.06', '\mu_{2} = 0.08', '\mu_{1} = \mu_{2} = 0.1','\mu_{1} = \mu_{2} = 0.02');
% %legend('\mu_{1} = 0.02', '\mu_{1} = 0.04', '\mu_{1} = 0.06', '\mu_{1} = 0.08', '\mu_{1} = \mu_{2} = 0.1', '\mu_{1} = \mu_{2} = 0.02');
% %legend('\mu_{2} = -0.02', '\mu_{2} = - 0.06', '\mu_{2} = - 0.1', '\mu_{2} = - 0.14', '\mu_{2} = - 0.18');
% legend('\mu_{2} = 0', '\mu_{2} = - 0.05', '\mu_{2} = - 0.1', '\mu_{2} = - 0.15', '\mu_{2} = - 0.2');
% %legend('\theta = 10',  '\theta = 15','\theta = 20');
% set(gca,'FontSize',fs);
% %print -r300 -deps2c prob_mu1_fix_mu2_smaller_w
% %print -r300 -deps2c prob_mu2_smaller_mu1_fix_w
% print -r300 -deps2c prob_mu1_fix1_mu2_neg05_w
% %print -r300 -deps2c prob_mu1_mu2_fix_mx3
% 
% 
% figure(2)
% plot(etA(:,1:4),'Linewidth', lw);
% hold on;
% %plot(etA(:,5), 'k-', 'Linewidth', lw);
% %hold on;
% % plot(x,eAs(1,:)+0*x, 'k-*', 'Linewidth', lw);
% %hold on;
% plot(etB(:,1:4),'--','Linewidth', lw);
% %hold on;
% %plot(x,eAs(2,:)+0*x, 'k-', 'Linewidth', lw)
% hold off;
% axis([0 nw 40 120]);
% set(gca, 'XTick', xt, 'FontSize', fs, 'FontName', fn);
% set(gca, 'XTickLabel', xL,'FontSize', fs,'FontName', fn);
% set(gca, 'YTick', yt, 'FontSize', fs, 'FontName', fn);
% set(gca, 'YTickLabel', yL,'FontSize', fs,'FontName', fn);
% %set(gca, 'YTickLabel', [],'FontSize', fs,'FontName', fn);
% xlabel('Attention switching w_{12}','FontSize',fs, 'FontName', fn);
% %set(gca, 'XTick', xt, 'FontSize', fs, 'FontName', fn);
% %set(gca, 'XTickLabel', xL,'FontSize', fs,'FontName', fn);
% ylabel('Mean choice time (arbitrary units)','FontSize',fs, 'FontName', fn);
% %legend('\mu_{2} = 0.02', '\mu_{2} = 0.04', '\mu_{2} = 0.06', '\mu_{2} = 0.08', '\mu_{1} = \mu_{2} = 0.1', '\mu_{1} = \mu_{2} = 0.02');
% %legend('\mu_{1} = 0.02', '\mu_{1} = 0.04', '\mu_{1} = 0.06', '\mu_{1} = 0.08', '\mu_{1} = \mu_{2} = 0.1','\mu_{1} = \mu_{2} = 0.02');
% %legend('\mu_{2} = -0.1', '\mu_{2} = -0.12', '\mu_{2} = -0.14', '\mu_{2} = -0.16', '\mu_{2} = -0.18','\mu_{2} = -0.2');
% %legend('\theta = 10', '\theta = 12', '\theta = 14', '\theta = 16', '\theta = 18','\theta = 20');
% %legend('\theta = 10',  '\theta = 15','\theta = 20');
% set(gca,'FontSize',fs);
% %print -r300 -deps2c et_mu1_fix_mu2_smaller_w
% %print -r300 -deps2c et_mu2_smaller_mu1_fix_w
% print -r300 -deps2c et_mu1_fix1_mu2_neg05_w
% %print -r300 -deps2c et_mu1_mu2_fix_mx3
% 
% %This example shows how to create two x axes with different scales (units)
%This technique can be adapted to create two y axes

% 
