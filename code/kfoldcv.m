function idx=kfoldcv(y,K)

idx=zeros(size(y));
label_set=unique(y);
m=length(label_set);

for i=1:m
    idx_sub=(y==label_set(i));
    idx_cv_sub=crossvalind('Kfold', sum(idx_sub), K);
    idx(idx_sub)=idx_cv_sub;
end

% for i=1:K
%     for j=i+1:K
%         if sum(idx==i)-sum(idx==j)>=2
%             idxi=(idx==i);
%             idx(idxi(1))=j;
%         elseif sum(idx==j)-sum(idx==i)>=2
%             idxj=(idx==j);
%             idx(idxj(1))=i;
%         end
%     end
% end
