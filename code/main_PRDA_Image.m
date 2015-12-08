%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PRDA
clc;
clear;
format compact;

srcStr = {'amazon','amazon','amazon','Caltech10','Caltech10','Caltech10','webcam','webcam','webcam'};
tgtStr = {'Caltech10','dslr','webcam','amazon','dslr','webcam','amazon','Caltech10','dslr'};

load(['../data/yt0All.mat']);
load(['../data/label_t0All.mat']);
for iData=1:9

    
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    % Preprocess data using Z-score
    load(['../data/' src '_SURF_L10.mat']);
    %     Xs=fts;
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xs = zscore(fts,1);
    Ys = labels;
    load(['../data/' tgt '_SURF_L10.mat']);
    %     Xt=fts;
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xt = zscore(fts,1);
    Yt = labels;
    
  
    yt0=yt0All{iData};
    label_t0=label_t0All{iData};
    options.NN=5;
    options.PCA=0;
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%% linear
     [label_t,predict_t,W,Accuracy(:,iData)]=DA_train_MKL(Xs,Ys,Xt,Yt,options,yt0,label_t0);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% kernel
  % [label_t,predict_t,Accuracy(:,iData)]=HD_DA_train_MKL(Xs,Ys,Xt,Yt,options,yt0,label_t0);
    
    clear yt0;
    clear label_t0;
    Accuracy
    
end
Accuracy
mean(Accuracy,2)


