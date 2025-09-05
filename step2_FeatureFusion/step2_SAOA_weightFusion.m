% 清空并载入数据集
clc; clear
load ('CD.mat');
load ('circRNA_CFS.mat');   
load ('MeSHSemanticSimilarity.mat');
load ('circRNA_nmf.mat');
load ('disease_nmf.mat');
load ('circRNA_ges.mat');
load ('disease_ges.mat');
% load ('circRNA_gipk.mat');
% load ('disease_gipk.mat');
load ('circRNA_mdmf.mat');
load ('disease_mdmf.mat');
% load ('circRNA_jaccard.mat');
% load ('disease_jaccard.mat');

%开始计时
tic;
% 记录初始内存状态
[usr_init, ~] = memory;

% input测试数据（1×N单元格数组）
features_c = cell(1,4);
features_d = cell(1,4);
% circRNA 
features_c{1} = circRNA_CFS;  % 特征矩阵1
features_c{2} = circRNA_nmf;  % 特征矩阵2
features_c{3} = circRNA_ges;  % 特征矩阵3
% features_c{3} = circRNA_gipk;  % 特征矩阵4
features_c{4} = circRNA_mdmf;  % 特征矩阵5
% disease
features_d{1} = MeSHSemanticSimilarity;  % 特征矩阵1
features_d{2} = disease_nmf;  % 特征矩阵2
features_d{3} = disease_ges;  % 特征矩阵3
% features_d{3} = disease_gipk;  % 特征矩阵4
features_d{4} = disease_mdmf;  % 特征矩阵5


%% 执行融合（加权融合 + 信息量最大化）
C_fused = weighted_feature_fusion(features_c);
D_fused = weighted_feature_fusion(features_d);

%% save

%结束计时
toc;
% 记录结束后的内存状态
[usr_end, ~] = memory;
% 计算内存消耗增量
memory_consumed = usr_end.MemUsedMATLAB - usr_init.MemUsedMATLAB;
fprintf('优化融合模块消耗了约 %.2f MB 内存\n', memory_consumed / 1024^2);
%% ========== 主函数：加权特征融合（信息量最大化） ==========
function fused_matrix = weighted_feature_fusion(features)
% 输入参数：
%   features: 1×N单元格数组，每个元素为二维特征矩阵
% 输出：
%   fused_matrix: 融合后的优质特征矩阵

% ========== 输入验证与预处理 ==========
validateattributes(features, {'cell'}, {'vector', 'nonempty'});
N = numel(features);

% 检查所有矩阵尺寸一致性
base_size = size(features{1});
for i = 2:N
    if ~isequal(size(features{i}), base_size)
        error('所有特征矩阵必须具有相同尺寸');
    end
end

% ========== 定义目标函数（最大化信息熵） ==========
fobj = @(weights) arrayfun(@(i) -compute_information(weights(i,:), features), (1:size(weights,1))');

% ========== 设置SAOA算法参数 ==========
pop_size = 50;      % 种群规模
max_iter = 200;     % 最大迭代次数
lb = zeros(1, N);   % 权重下界
ub = ones(1, N);    % 权重上界
dim = N;            % 优化变量维度

% ========== 运行SAOA优化算法 ==========
[gbest_fit,gbest, ~] = SAOA(pop_size, max_iter, lb, ub, dim, fobj);

% ========== 后处理与融合 ==========
optimal_weights = gbest ./ sum(gbest);  % 归一化权重
fused_matrix = weighted_sum(features, optimal_weights);
end

%% ========== 辅助函数：加权求和 ==========
function result = weighted_sum(features, weights)
% 输入验证
if numel(features) ~= numel(weights)
    error('权重数量与特征矩阵数量不匹配');
end

% 初始化结果矩阵
result = zeros(size(features{1}));

% 加权求和
for i = 1:numel(features)
    result = result + weights(i) * features{i};
end
end

%% ========== 辅助函数：计算信息量（熵） ==========
function info = compute_information(w, features)
% 权重归一化处理
w_normalized = w ./ sum(w);

% 特征矩阵融合
fused = weighted_sum(features, w_normalized);

% 计算信息熵（最大化目标）
fused_normalized = fused ./ sum(fused(:)); % 归一化为概率分布
info = -sum(fused_normalized(:) .* log2(fused_normalized(:) + eps)); % 避免log(0)
end