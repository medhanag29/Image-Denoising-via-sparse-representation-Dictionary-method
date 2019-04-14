%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file tests our method for image deblurring
clear; close all;
     rng(2013,'v5uniform'); rng(2013,'v5normal');

% load pre-learned dictionaries
load LearnDict.mat;

 n = size(Dl_lam_08,1);

 D = [ones(n,1),Dl_lam_08];

load X.mat;

 K = size(D,2);
 [N1,N2] = size(X);

n1 = 8; n2 =8 ;

 StPt = [1 1; 1 5; 5 1];
 StPt_num = 3;
 Xrec = cell(1,StPt_num);

blur_kernel = fspecial('average',[9,9]);

 [t,s] = size(blur_kernel);
 t = floor(t/2); s = floor(s/2);
 v = zeros(N1,N2);
v(floor(N1/2)+1-t:floor(N1/2)+1+t,floor(N2/2)+1-s:floor(N2/2)+1+s) = blur_kernel;
V = psf2otf(v);

b = ifft2(V.*(fft2(X))); b = b(:);

nois = randn(N1*N2,1);

% add 1% noise
sig = 0.01*max(abs(b))/max(abs(nois));
b = b+sig*nois;
p=[];w1=[];
for w=0.1:0.1:50
    
for num = 1:StPt_num
    
    rowSt = StPt(num,1); colSt = StPt(num,2);
    
    sgn1 = 0; sgn2 = 0;
    
    rowN = floor((N1-rowSt+1)/n1); colN = floor((N2-colSt+1)/n2);
    res1 = N1-rowSt+1-rowN*n1; res2 = N2-colSt+1-colN*n2;
    if res1 > 0        sgn1 = 1;      end
    if res2 > 0        sgn2 = 1;      end
    q = rowN*colN+sgn1*colN+sgn2*rowN+sgn1*sgn2;
    if rowSt > 1
        q = q + colN + sgn2;
    end
    
    if colSt > 1
        q = q + rowN + sgn1;
    end
    
    A = BlurAoperator(blur_kernel,D,n1,n2,N1,N2,rowSt,colSt,q);
    
    opts = []; opts.maxit = 1000; opts.tol = 1e-4;
    opts.rho = 0.1*sig;
    opts.x0 = randn(K*q,1);
    opts.weights = [zeros(1,q);ones(K-1,q)];
    opts.weights = w*opts.weights(:);
    x = yall1(A,b,opts);
    Xrec{num} = vPtRtFunc(D*reshape(x,K,q),n1,n2,N1,N2,rowSt,colSt);
end

% taking average over three recovered images
Xav = zeros(N1,N2);
for i = 1:StPt_num
    Xav = Xav + Xrec{i};
end
Xav = Xav/StPt_num;
psnr = measerr(X,Xav,max(X(:)));
if psnr>=25
    p=[p psnr]
    
    w1=[w1,w]
 else
 return;
end
end
imshow(Xav);
disp(psnr);