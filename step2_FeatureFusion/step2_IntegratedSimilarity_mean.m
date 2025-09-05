% 清空并载入数据集
clc; clear
load ('CD.mat');
load ('circRNA_CFS.mat');   
load ('MeSHSemanticSimilarity.mat');
load ('circRNA_nmf.mat');
load ('disease_nmf.mat');
load ('circRNA_ges.mat');
load ('disease_ges.mat');
load ('circRNA_gipk.mat');
load ('disease_gipk.mat');
% load ('circRNA_mdmf.mat');
% load ('disease_mdmf.mat');


% input测试数据（1×N单元格数组）
features_c = cell(1,4);
features_d = cell(1,4);
% circRNA 
features_c{1} = circRNA_CFS;  % 特征矩阵1
features_c{2} = circRNA_nmf;  % 特征矩阵2
features_c{3} = circRNA_ges;  % 特征矩阵3
features_c{4} = circRNA_gipk;  % 特征矩阵4
% features_c{5} = circRNA_mdmf;  % 特征矩阵5
% features_c{6} = circRNA_jaccard;  % 特征矩阵6
% disease
features_d{1} = MeSHSemanticSimilarity;  % 特征矩阵1
features_d{2} = disease_nmf;  % 特征矩阵2
features_d{3} = disease_ges;  % 特征矩阵3
features_d{4} = disease_gipk;  % 特征矩阵4
% features_d{5} = disease_mdmf;  % 特征矩阵5
% features_d{6} = disease_jaccard;  % 特征矩阵6

%% Part:1 circRNA integrated similarities
n = size(features_c,2);
% 计算每个矩阵的和
sum_c = sum(cat(3, features_c{:}), 3); % 沿第3维求和(先拼接在求和)
circ_feature = sum_c/n;  
C_fused = circ_feature;

%% Part:2 disease integrated similarities 
n = size(features_d,2);
sum_d = sum(cat(3, features_d{:}), 3); % 沿第3维求和(先拼接在求和)
dis_feature = sum_d/n; 
D_fused = dis_feature;

%% SAVE DATE
