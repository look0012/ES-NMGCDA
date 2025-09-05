 function [Prediction,Accuracy,Dec_values]=SVM(test_data,test_lab,train_data,train_lab,KFold)

    Dec_values{1,KFold}=[];
    Prediction{1,KFold}=[];
    Accuracy{1,KFold}=[];
for i=1:KFold
    testdata=test_data{i,1};
    test1ab=test_lab{i,1};
    traindata=train_data{i,1};
    train1ab=train_lab{i,1};
     %%%%%SVM  Model%%%%
%     model{i,1}=fitcsvm(traindata,train1ab,'BoxConstraint',8,'KernelFunction','rbf','KernelScale',0.9);
    model{i,1}=fitcsvm(traindata,train1ab,'BoxConstraint',2,'KernelFunction','rbf','KernelScale',0.3);
    [prediction,dec_values]=predict(model{i,1},testdata); %dec_values第1列：样本属于负类的得分,第2列：样本属于正类的得分。

    Dec_values{1,i}=dec_values;
    Prediction{1,i}=prediction;
    accnum=0;
    for k=1:size(test1ab,1)
        %%%%%%%折
        if(test1ab(k,1)==prediction(k,1))
        accnum=accnum+1;
        end
    end
    accnum/size(test1ab,1)
    Accuracy{1,i}=accnum/size(test1ab,1);
end
end