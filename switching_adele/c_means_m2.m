function [proba, probb, eta, etb] = c_means_m2(para,tau)

%---------------------------------------------



r12=0;  %attention switching from attribute 1 to attribute 2


r21 = para(5); %attention switching from attribute 2 to attribute 1


r11=1-r12;  %Probability staying at attribute 1
r22=1-r21;  %Probability staying at attribute 2

%--------------------------------------------------------------------------
%Constructing the matrix
%--------------------------------------------------------------------------

alpha =1; 
%tau=.1; %scaling time unit
%tau=.25;
sigma=1;
delta=alpha*sigma*sqrt(tau);  % step size
m=2*round(para(6)/delta)+1;%Matrix size

x = -(m-1)/2:(m-1)/2; %states

%--------------------------------------------------------------------------
%drifts

s1=para(3); %or para(x2) for OUP for first process
s2=para(3) ; %or para(x3) for OUP for second process

mux1=para(1)-s1*delta*x;
mux2=para(2)-s2*delta*x;

        %------------------------------------------------------------------
        
        %------------------------------------------------------------------
        %Transition probabilities
        
        px1 = 1/(2*alpha)*(1-mux1*sqrt(tau)/sigma);
        qx1 = 1/(2*alpha)*(1+mux1*sqrt(tau)/sigma);

        px2 = 1/(2*alpha)*(1-mux2*sqrt(tau)/sigma);
        qx2 = 1/(2*alpha)*(1+mux2*sqrt(tau)/sigma);
        pq= 1-(1/alpha)-0*x;

        %--------------------------------------------------------------------------
        %Transition matrix

        tm1 = (zeros(m,m)); %transition matrix for attribute 1
        tm2 = (zeros(m,m)); %transition matrix for attribute 2

        tm1(1,1)=1; tm2(1,1)=1;
        tm1(m,m)=1; tm2(m,m)=1;

        

        i=1;

        while i <= m-2,
            i=i+1;
            j=[(i-1) i (i+1)];
            tm1(i,j)=[r11*px1(i) r11*pq(i) r11*qx1(i)];
            tm2(i,j)=[r22*px2(i) r22*pq(i) r22*qx2(i)];
        end

        
        
             
        %------------------------------------------------------------------
        %Combining the transition matrix tm  with the r matrices

        n_tm=size(tm1,1);  

        %id = sparse(1:n_tm,1:n_tm,1);
        id=eye(n_tm);
        
        d12=r12*id;
        d21=r21*id;

        d12(1,1)=0; d12(m,m)=0;
        d21(1,1)=0; d21(m,m)=0;

        tm1=[tm1 d12];
        tm2=[d21 tm2];
        tm=[tm1;tm2];
        %------------------------------------------------------------------
        %Rearranging the MAtrix 
        %R-Vectors and Q-Matrix

        per1=[tm(:,1) tm(:,m+1) tm(:,2:m-1) tm(:,m+2:2*m-1) tm(:,m) tm(:,2*m)];
        per2=[per1(1,:);per1(m+1,:);per1(2:m-1,:);per1(m+2:2*m-1,:);per1(m,:);per1(2*m,:)];

        rb=per2(3:2*m-2,1:2);
        ra=per2(3:2*m-2,2*m-1:2*m);  %lower left part of the matrix
        q=per2(3:2*m-2,3:2*m-2);

        n=size(q,1);

        id = sparse(1:n,1:n,1);

        invq =inv(id-q);

        %------------------------------------------------------------------
        % prob to consider a certain attribute first is 1

        z1=(zeros(n/2,1));
        z2=(zeros(n/2,1));
        nn=(n/2-1)/2+1;
        z2(nn,1)=1; % Start with attribute 2 
        z=[z1;z2];

        
        %------------------------------------------------------------------
        % Choice probabilities for A and B

        proba = z'*invq*ra;
        probb = z'*invq*rb;

       

        %------------------------------------------------------------------ 
        %Expected value for A and B, not conditioned yet
        
        eta =tau*z'*(invq*invq)*ra;
        etb =tau*z'*(invq*invq)*rb;
        
 
eta=sum(eta);
etb=sum(etb);
proba=sum(proba);
probb=sum(probb);       

eta=eta./proba;
etb=etb./probb;
            
   







