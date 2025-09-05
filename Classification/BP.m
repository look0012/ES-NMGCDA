 function [Prediction,Accuracy,Dec_values]=BP(test_data,test_lab,train_data,train_lab,KFold)

    Dec_values{1,KFold}=[];
    Prediction{1,KFold}=[];
    Accuracy{1,KFold}=[];
for i=1:KFold
    testdata=test_data{i,1};
    test1ab=test_lab{i,1};
    traindata=train_data{i,1};
    train1ab=train_lab{i,1};
    %%%%BP Model%%%
    model{i,1} = newff(traindata',train1ab',[4,4,3],{'tansig','purelin'},'trainlm');
    model{i,1}.trainParam.epochs = 300; %训练次数
    model{i,1}.trainParam.goal = 0; %训练目标最小误差
    model{i,1}.trainParam.lr = 0.08;    %学习率
    %view(model{i,1})
    %%%神经网络输入列向量是输入样本
    model{i,1} = train(model{i,1},traindata',train1ab');
    out = sim(model{i,1},testdata');
%     [prediction,dec_values]=predict(model{i,1},testdata');
%     prediction_out=vec2ind(out);
    prediction=out';
    prediction(prediction>=0.37)=1;
    prediction(prediction<0.37)=-1;
    dec_values2=prediction;
    dec_values2(dec_values2==-1)=0;
    dec_values1=prediction;
    dec_values1(dec_values1==1)=0;
    dec_values1(dec_values1==-1)=1;
    dec_values = [dec_values1,dec_values2];
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