function [S,out_t,idx_t,acc_ds]=domain_separator_train_MPM(Xs,Xt,options)

Yds=[ones(size(Xt,1),1);-ones(size(Xs,1),1)];
Xds=[Xt;Xs];
% Model selection by K-fold CV
Kfold=options.Kfold;
idx=kfoldcv(Yds,Kfold);
acc=[];
for fold=1:Kfold
    xte=Xds(idx==fold,:);
    yte=Yds(idx==fold);
    
    xtr=Xds(idx~=fold,:);
    ytr=Yds(idx~=fold);
    xtrXt=xtr(ytr==1,:);
    xtrXs=xtr(ytr==-1,:);
    xtrXt_mean=mean(xtrXt);
    xtrXs_mean=mean(xtrXs);
    CovxtrXt=cov(xtrXt);
    CovxtrXs=cov(xtrXs);
    
    %%%% parameters for MPM
    nu=0;
    rho_x=10.^[-4:3];
    rho_y=10.^[-4:3];
    for rho=1:length(rho_x)
        gauss_assump=0;
        algoparam=-1;
        tol=-1;
        maxiter=-1;
        %%%%MPM
        [alfa,a,b] = build_robMPM_lin_binclass_LSreg(xtrXt_mean',xtrXs_mean',CovxtrXt',CovxtrXs',nu,rho_x(rho),rho_y(rho),gauss_assump,algoparam,tol,maxiter);
        
        out=sign(xte*a-b);
        acc(rho,fold)=mean(out==yte);
    end
    
end
acc_mean=mean(acc,2);
[row,col]=find(acc_mean==max(max(acc_mean)));
rho_best=rho_x(row(1,1))
rho_x=rho_best;
rho_y=rho_best;

XtMean=mean(Xt);
XsMean=mean(Xs);
CovXt=cov(Xt);
CovXs=cov(Xs);
%%%MPM
[alfa,abest,bbest] = build_robMPM_lin_binclass_LSreg(XtMean',XsMean',CovXt',CovXs',nu,rho_best,rho_best,gauss_assump,algoparam,tol,maxiter);
out_all=Xds*abest-bbest;
% [sort_t,idx_t]=sort(out_all(1:size(Xt,1)));
out_t=out_all(1:size(Xt,1));
[sort_t, idx_t]=sort(out_all(1:size(Xt,1)));
acc_ds=mean(sign(out_all)==Yds)
A=options.A;
%%%%%%
% S=1./(1+exp(A*(out_t)));
%%%%%%
S=1./exp(A*(out_t));
%%%%%%

disp(['Domain separator accuracy: ',num2str(acc_ds),'%']);