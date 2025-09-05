function [Prediction, Accuracy, Dec_values] = CF(test_data, test_lab, train_data, train_lab, K)
% Causal Forest 因果森林
% 输入:
%   test_data - K×1 cell array，每cell包含测试数据(样本×特征)
%   test_lab - K×1 cell array，每cell包含测试标签
%   train_data - K×1 cell array，每cell包含训练数据
%   train_lab - K×1 cell array，每cell包含训练标签
%   K - 交叉验证折数(应与cell数组长度一致)
%
% 输出:
%   Prediction - 1×K cell array，每cell包含预测标签
%   Accuracy - K×1向量，每折的准确率
%   Dec_values - 1×K cell array，每cell包含预测得分

% 参数校验
if length(test_data) ~= K || length(train_data) ~= K
    error('输入cell数组长度与K值不匹配');
end

% 初始化输出
Prediction = cell(1,K);
Accuracy = zeros(K, 1);
Dec_values = cell(1,K);

% 并行处理每折数据
for k = 1:K
    % 获取当前折数据
    curr_train_data = train_data{k};
    curr_train_lab = train_lab{k};
    curr_test_data = test_data{k};
    curr_test_lab = test_lab{k};
    
    % 训练因果森林模型
%     num_trees = 200; % 每折的树数量
    num_trees = 30; % 每折的树数量
    num_features = size(curr_train_data, 2);
    mtry = max(floor(sqrt(num_features)), 1);
    
    % 初始化存储
    all_predictions = zeros(size(curr_test_data, 1), num_trees);
    
    % 训练多棵树
    for t = 1:num_trees
        % 自助采样
        sample_idx = randsample(size(curr_train_data, 1), size(curr_train_data, 1), true);
        X_boot = curr_train_data(sample_idx, :);
        Y_boot = curr_train_lab(sample_idx);
        
        % 构建因果树
%         tree = fitctree(X_boot, Y_boot, ...
%             'PredictorSelection', 'curvature', ...
%             'MaxNumSplits', 100, ...
%             'MinLeafSize', 0.5, ...
%             'SplitCriterion', 'deviance', ...
%             'NumVariablesToSample', mtry);
        tree = fitctree(X_boot, Y_boot, ...
            'PredictorSelection', 'curvature', ...
            'MaxNumSplits', 50, ...
            'MinLeafSize', 0.5, ...
            'SplitCriterion', 'deviance', ...
            'NumVariablesToSample', mtry);
        
        % 预测
        [pred, scores] = predict(tree, curr_test_data);
        all_predictions(:, t) = pred;
    end
    
    % 聚合预测结果
    final_pred = mode(all_predictions, 2); % 投票法
    Prediction{1,k} = final_pred;
    
    % 计算准确率
    Accuracy(k) = mean(final_pred == curr_test_lab);
    
    % 计算预测得分(平均概率)
%     Dec_values{k} = mean(all_predictions, 2);
    Dec_values{1,k} = scores;
end

% 显示总体性能
disp(['平均交叉验证准确率: ', num2str(mean(Accuracy))]);
end