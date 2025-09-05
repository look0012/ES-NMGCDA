 % 双向学习
function [Prediction, Accuracy, Dec_values] = BL(test_data, test_lab, train_data, train_lab, K)
    % 初始化输出变量
    Prediction = cell(1, K);
    Accuracy = zeros(K, 1);
    Dec_values = cell(1, K);
    
    % 检查输入数据一致性
    if length(test_data) ~= K || length(test_lab) ~= K || ...
       length(train_data) ~= K || length(train_lab) ~= K
        error('输入数据的K折数与指定的K值不匹配');
    end
    
    % 对每一折进行训练和测试
    for fold = 1:K
        fprintf('\nProcessing fold %d/%d...\n', fold, K);
        
        % 获取当前折的数据
        XTrain = train_data{fold};
        YTrain = train_lab{fold};
        XTest = test_data{fold};
        YTest = test_lab{fold};
        
        % 双向学习策略
        % 前向学习路径
        fprintf('Training forward learner...\n');
        forwardModel = trainForwardLearner(XTrain, YTrain);
        
        % 后向学习路径
        fprintf('Training backward learner...\n');
        backwardModel = trainBackwardLearner(XTrain, YTrain);
        
        % 特征融合
        fprintf('Fusing features...\n');
        [fusedTrain, fusedTest] = fuseFeatures(XTrain, XTest, forwardModel, backwardModel);
        
        % 元学习器训练 - 修改为使用Bag方法以支持并行
        fprintf('Training meta-learner...\n');
        finalModel = fitcensemble(fusedTrain, YTrain, ...
                                 'Method', 'Bag', ...  % 改为Bag方法以支持并行
                                 'NumLearningCycles', 100, ...
                                 'Learners', 'tree', ...
                                 'Options', statset('UseParallel', true));
        
        % 预测
        [YPred, scores] = predict(finalModel, fusedTest);
        
        % 存储结果
        Prediction{fold} = YPred;
        Accuracy(fold) = sum(YPred == YTest) / length(YTest);
        Dec_values{fold} = scores(:,2); % 正类的得分
        
        fprintf('Fold %d accuracy: %.2f%%\n', fold, Accuracy(fold)*100);
    end
end

%% 前向学习器
function model = trainForwardLearner(X, Y)
    % 使用随机森林计算特征重要性
    if isa(X, 'double')
        X = array2table(X);
    end
    
    tree = TreeBagger(50, X, Y, 'Method', 'classification', 'OOBPredictorImportance', 'on');
    imp = tree.OOBPermutedPredictorDeltaError;
    
    % 选择重要性最高的30个特征
    [~, idx] = sort(imp, 'descend');
    numFeaturesToSelect = min(30, size(X,2));
    selectedIdx = idx(1:numFeaturesToSelect);
    
    % 训练SVM分类器
    svmModel = fitcsvm(X(:, selectedIdx), Y, ...
                      'KernelFunction', 'rbf', ...
                      'Standardize', true);
    
    model.selectedFeatures = selectedIdx;
    model.svm = svmModel;
end

%% 后向学习器
function model = trainBackwardLearner(X, Y)
    % 使用PCA进行降维
    [coeff, score, ~, ~, explained] = pca(X);
    
    % 选择解释95%方差的成分
    cumVar = cumsum(explained);
    numComponents = find(cumVar >= 95, 1, 'first');
    if isempty(numComponents)
        numComponents = size(X,2);
    end
    
    encoded = score(:, 1:numComponents);
    
    model.pcaCoeff = coeff;
    model.numComponents = numComponents;
    model.encodedFeatures = encoded;
end

%% 特征融合
function [fusedTrain, fusedTest] = fuseFeatures(XTrain, XTest, forwardModel, backwardModel)
    % 前向特征
    forwardTrain = XTrain(:, forwardModel.selectedFeatures);
    forwardTest = XTest(:, forwardModel.selectedFeatures);
    
    % 后向特征
    if istable(XTrain)
        XTrain = table2array(XTrain);
        XTest = table2array(XTest);
    end
    backwardTest = XTest * backwardModel.pcaCoeff(:, 1:backwardModel.numComponents);
    
    % 特征融合（拼接）
    fusedTrain = [forwardTrain, backwardModel.encodedFeatures];
    fusedTest = [forwardTest, backwardTest];
end