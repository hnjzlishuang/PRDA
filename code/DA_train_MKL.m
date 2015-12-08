% function [label_t,predict_t,W,Accuracy]=DA_train_MKL(Xs,Ys,Xt,Yt,options,sigmama,C,d)
function [label_t,predict_t,W,Accuracy,recordacc]=DA_train_MKL(Xs,Ys,Xt,Yt,options,yt0,label_t0)
Nt=length(Yt);
Ns=length(Ys);
class_set=unique(Ys);
nClass=length(class_set);

acc_t0=mean(yt0==Yt)*100;
%%%%Dimensionality_Reduction_Type可选：PCA，Laplacian，LLE
para.Dimensionality_Reduction_Type='PCA';

%% PCA
disp('para.Dimensionality_Reduction_Type...')
dim=100;
if strcmp(para.Dimensionality_Reduction_Type,'PCA')
    [pc,score,latent,tsquare]=princomp([Xs;Xt]);
    Xs=score(1:length(Ys),1:dim);
    Xt=score(length(Ys)+1:end,1:dim);
    disp(['Variance preserved',num2str(sum(latent(1:dim))/sum(latent))])
end

%%
% Compute graph Laplacian
disp('Compute graph Laplacian');
options.GraphWeights='binary';
options.GraphDistanceFunction='euclidean';
options.LaplacianNormalize=0;
options.LaplacianDegree=1;
L=laplacian(options,Xt);
%%
for iA=1
    options.A=1;
    options.Kfold=2;
    
    %Train domain separator
    disp('Train domain separator ...');
    S=zeros(Nt,1);
    for class=1:nClass
        Ntc=sum(yt0==class_set(class));
        %%%%%%%%%%%%%%%%%%%%%   MPM
        [S_tmp,out_all,idx_t,acc_ds]=domain_separator_train_MPM(Xs(Ys==class_set(class),:),Xt,options);
        
        %     [S_tmp,dis]=domain_correspondence(Xs(Ys==class_set(class),:),Xt);
        S=S+S_tmp.*(yt0==class_set(class))*(Nt/Ntc);
        
    end
    %% 由于source domain和target domain 都是均值为0的，故要用每一类s来衡量
    S_source=zeros(Ns,1);
    for class=1:nClass
        % choose the domain separator & reweighting scheme
        %%%%%%%%%%%%%%%%%%%%
        %[S_tmp,out_all,idx_t,acc_ds]=dormain_separator_train(Xs(Ys==class_set(class),:),Xt,options);
        %%%%%%%%%%%%%%%%%%%%%%
        % [S_tmp,out_all,idx_t,acc_ds]=dormain_separator_train_theory(Xs(Ys==class_set(class),:),Xt,options);
        %%%%%%%%%%%%%%%%%%%%%   MPM
        [S_s,~,~,~]=domain_separator_train_MPM(Xt,Xs(Ys==class_set(class),:),options);
        
        S_source(find(Ys==class_set(class)))=S_s;
    end
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Train DA classifier
    for i=1:7
        %       options.lambda=10^(i-7);
        options.lambda=10^(i);
        if size(Xt,1)<size(Xt,2)
            W=((diag(S)+options.lambda*L)*Xt)\bsxfun(@times,S,label_t0);
        else
            W=(Xt'*(diag(S)+options.lambda*L)*Xt+0.01*eye(size(Xt,2)))\(Xt'*bsxfun(@times,S,label_t0));
        end
        
        label_t=Xt*W;
        label_s=Xs*W;

        if nClass==2
            predict_t=sign(label_t);
            predict_s=sign(label_s);
        else
            [~,predict_t] = max(label_t,[],2);
            [~,predict_s] = max(label_s,[],2);
        end
        acc(i,1)=mean(predict_t==Yt);
        acc_v(i,1)=S_source'*(predict_s==Ys);
        PPT(:,i)=predict_t;
    end
    
    recordacc(:,iA)=acc;
    recordacc_v(:,iA)=acc_v;
    acc
    acc_v
    acc_best=100*max(max(acc));
    Ind=find(( acc_v==max(acc_v(:,1)))==1);
    acc_da=100*acc(Ind(end));
    yt0=PPT(:,Ind(end));
    %%%%%%%%%%%%%MKL for target domain
    Numclass=unique(yt0);
    label_t0=zeros(size(Xt,1),length(Numclass));
    for i=1:Nt
        label_t0(i,yt0(i))=1;
    end
end

recordacc
recordacc_v
Indmatrix=(recordacc_v==(max(max(recordacc_v))));
acc_da=recordacc(Indmatrix)*100;
Indvector=sum(Indmatrix);
Indcol=find(Indvector==1);
acc_best=max(acc)*100;

disp(['Accuracy with domain adaptation: ',num2str(acc_da(1,1)),'%/',...
    num2str(acc_best),'%(Best)/',num2str(acc_t0),'%(No adaption)']);
Accuracy=[acc_t0(1);acc_da(1,1);acc_best(1,1)];

%% Convert single column labels to multi-column labels
function label=singlelbs2multilabs(out,y,nclass,winner_take_all)
N=size(out,1);
label=zeros(N,10);
count=0;
for i=1:10
    for j=i+1:10
        count=count+1;
        label(:,i)=label(:,i)+(out(:,count)>0);
        label(:,j)=label(:,j)+(out(:,count)<0);
    end
end
[C,I] = max(label,[],2);
if winner_take_all
    tmp=sparse([1:N]',y,ones(N,1),N,nclass);
    label=full(tmp);
end