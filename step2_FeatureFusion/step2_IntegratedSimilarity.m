% Calculates integrated similarities of circRNA and disease.

% 清空并载入数据集
clc; clear
load ('CD.mat');
load ('circRNA_CFS.mat');   
load ('MeSHSemanticSimilarity.mat');
load ('circRNA_nmf.mat');
load ('disease_nmf.mat');

cd_adjmat = CD;

ciRNA_ss = circRNA_CFS; % circRNA CFS similarity
disease_ss = MeSHSemanticSimilarity; % MeSHSemanticSimilarity;
CC = circRNA_nmf; % circRNA Gaussian interaction kernel similarity
DD = disease_nmf; % disease Gaussian interaction kernel similarity

[nc,nd ]= size(cd_adjmat);

%% Part:1 circRNA integrated similarities
circ_feature = zeros(nc,nc);
for i = 1:nc
    for j = 1:nc
        if ciRNA_ss(i,j)~=0
            circ_feature(i,j) = ciRNA_ss(i,j); 
        else 
            circ_feature(i,j) = CC(i,j); 
        end
    end
end
C_fused = circ_feature;

%% Part:2 disease integrated similarities 
dis_feature = zeros(nd,nd);

for i = 1:nd
    for j = 1:nd
        if disease_ss(i,j)~=0
            dis_feature(i,j) = disease_ss(i,j); 
        else 
            dis_feature(i,j) = DD(i,j);  
        end
    end
end
D_fused = dis_feature;
%% SAVE DATE

save('./step2.1_FeatureFusion/CDfused_IntegratedSimilarity.mat','C_fused','D_fused');
