function [Prediction, Accuracy, Dec_values] = BT(test_data, test_lab, train_data, train_lab, K)
    % 初始化输出变量
    Prediction = cell(1, K);
    Accuracy = zeros(K, 1);
    Dec_values = cell(1, K);
    
    % 检查输入数据的一致性
    if length(test_data) ~= K || length(test_lab) ~= K || ...
       length(train_data) ~= K || length(train_lab) ~= K
        error('输入数据的K折数与指定的K值不匹配');
    end
    
    % 对每一折进行训练和测试
    for fold = 1:K
        fprintf('Processing fold %d/%d...\n', fold, K);
        
        % 获取当前折的训练和测试数据
        XTrain = train_data{fold};
        YTrain = train_lab{fold};
        XTest = test_data{fold};
        YTest = test_lab{fold};
        
        % 训练分类器 - 使用集成学习方法（Bagged Trees）
        rng(1); % 设置随机种子保证可重复性
        classifier = fitcensemble(XTrain, YTrain, ...
                                'Method', 'Bag', ...
                                'NumLearningCycles', 4, ...
                                'Learners', 'tree', ...
                                'ClassNames', [1; -1]);
        
        % 预测测试集
        [YPred, scores] = predict(classifier, XTest);
        
        % 存储结果
        Prediction{fold} = YPred;
        Accuracy(fold) = sum(YPred == YTest) / length(YTest);
        Dec_values{fold} = scores(:,2); % 正类的得分
        
        fprintf('Fold %d accuracy: %.2f%%\n', fold, Accuracy(fold)*100);
    end
end