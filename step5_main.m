data = [p_sample; n_sample]; 
% index = [name_p; name_n]; 


tic;
[usr_init, ~] = memory;

KFold=5;
[test_data,train_data,test_lab,train_lab] = KFoldCrossValidation(data,KFold);
% [test_index,test_data,train_data,test_lab,train_lab] = KFoldCrossValidation2(index,data,KFold);% K折交叉验证

[Prediction,Accuracy,Dec_values] = CF(test_data,test_lab,train_data,train_lab,KFold);


toc;
[usr_end, ~] = memory;

memory_consumed = usr_end.MemUsedMATLAB - usr_init.MemUsedMATLAB;
fprintf('CF模块消耗了约 %.2f MB 内存\n', memory_consumed / 1024^2);


[VACC,VSN,VPE,VMCC,VF1] = roc( Prediction,test_lab',KFold);
[VAUC]=plotroc(test_lab,Dec_values,KFold);

ValACC=mean(cell2mat(VACC));
ValPE=mean(cell2mat(VPE));
ValSN=mean(cell2mat(VSN));
ValMCC=mean(cell2mat(VMCC));
ValF1=mean(cell2mat(VF1));
ValAUC=mean(VAUC);

stdACC=std(cell2mat(VACC));
stdPE=std(cell2mat(VPE));
stdSN=std(cell2mat(VSN));
stdMCC=std(cell2mat(VMCC));
stdF1=std(cell2mat(VF1));
stdAUC=std(VAUC);

