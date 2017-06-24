clc
clear
% 1.1 Load the data
load('HW1Dataset.mat')

% 1.2 Normalize data
norm_data = normc(data);

% 1.3.1 Cross_validatin_Index
Indices = crossvalind('Kfold', length(norm_data), 10);
% 1.3.2 change label to 1 & 2
new_labels = labels + 1;

% Linear Regression
% correrate_train = 0;
% sensitivity_train = 0;
% specificity_train = 0;
% correrate_test = 0;
% sensitivity_test = 0;
% specificity_test = 0;
% 
% for i = 1:10
%     % Training Set
%     X = norm_data(Indices == i,:);
%     Y = new_labels(Indices == i,:);
%     len_data = length(X);
%     train_len = double(len_data - int16(len_data/10));
%     test_len = double(int16(len_data/10));
%     % modeling
%     B = mnrfit(X(1:train_len,:),Y(1:train_len,:));
%     % apply model on training 
%     class_out = mnrval(B,X(1:train_len,:));
%     % compute the confusion matrix
%     Y_pred = (class_out(:,1) < class_out(:,2));
%     Y_pred = Y_pred + 1;
%     result_training = classperf(Y(1:train_len,:),Y_pred);
%     correrate_train= correrate_train + result_training.CorrectRate;
%     sensitivity_train = sensitivity_train + result_training.Sensitivity;
%     specificity_train = specificity_train + result_training.Specificity;
%     
%     % 1.4 Apply model on Testing Set
%     class_out_test = mnrval(B,X(train_len+1:end,:));
%     Y_pred_test = (class_out_test(:,1) < class_out_test(:,2));
%     Y_pred_test = Y_pred_test + 1;
%     [correrate_test,sensitivity_test, specificity_test] = cfm_test(Y_pred_test,Y,train_len,test_len,correrate_test,sensitivity_test,specificity_test);
% end
% 
% [ann_avg_accy_train,ann_avg_sens_train,ann_avg_spec_train,ann_avg_accy_test,ann_avg_sens_test,ann_avg_spec_test] = avg_eval_matx(correrate_train, sensitivity_train, specificity_train, correrate_test, sensitivity_test, specificity_test);



% % Gaussian Naive Bayes 
% correrate_train = 0;
% sensitivity_train = 0;
% specificity_train = 0;
% correrate_test = 0;
% sensitivity_test = 0;
% specificity_test = 0;
% for i = 1:10
%     % Training Set
%     X = norm_data(Indices == i,:);
%     Y = new_labels(Indices == i,:);
%     len_data = length(X);
%     train_len = double(len_data - int16(len_data/10));
%     test_len = double(int16(len_data/10));
%     % modeling
%     B = fitcnb(X(1:train_len,:),Y(1:train_len,:));
%     % apply model on training 
%     Y_pred = predict(B,X(1:train_len,:));
%     % compute the confusion matrix
%     result_training = classperf(Y(1:train_len,:),Y_pred);
%     correrate_train= correrate_train + result_training.CorrectRate;
%     sensitivity_train = sensitivity_train + result_training.Sensitivity;
%     specificity_train = specificity_train + result_training.Specificity;
%     
%     % 1.4 Apply model on Testing Set
%     Y_pred_test = predict(B,X(train_len+1:end,:));
%     [correrate_test,sensitivity_test, specificity_test] = cfm_test(Y_pred_test,Y,train_len,test_len,correrate_test,sensitivity_test,specificity_test);
% end
% 
% [ann_avg_accy_train,ann_avg_sens_train,ann_avg_spec_train,ann_avg_accy_test,ann_avg_sens_test,ann_avg_spec_test] = avg_eval_matx(correrate_train, sensitivity_train, specificity_train, correrate_test, sensitivity_test, specificity_test);


% Linear Discriminant

% correrate_train = 0;
% sensitivity_train = 0;
% specificity_train = 0;
% correrate_test = 0;
% sensitivity_test = 0;
% specificity_test = 0;
% 
% for i = 1:10
%     % Training Set
%     X = norm_data(Indices == i,:);
%     Y = new_labels(Indices == i,:);
%     len_data = length(X);
%     train_len = double(len_data - int16(len_data/10));
%     test_len = double(int16(len_data/10));
%     % modeling
%     lda = fitcdiscr(X(1:train_len,:),Y(1:train_len,:));
%     
%     % apply model on training 
%     Y_pred = predict(lda,X(1:train_len,:));
%     % compute the confusion matrix
%     result_training = classperf(Y(1:train_len,:),Y_pred);
%     correrate_train= correrate_train + result_training.CorrectRate;
%     sensitivity_train = sensitivity_train + result_training.Sensitivity;
%     specificity_train = specificity_train + result_training.Specificity;
%     
%     % 1.4 Apply model on Testing Set
%     Y_pred_test = predict(lda,X(train_len+1:end,:));
%     [correrate_test,sensitivity_test, specificity_test] = cfm_test(Y_pred_test,Y,train_len,test_len,correrate_test,sensitivity_test,specificity_test);
% end
% 
% [ann_avg_accy_train,ann_avg_sens_train,ann_avg_spec_train,ann_avg_accy_test,ann_avg_sens_test,ann_avg_spec_test] = avg_eval_matx(correrate_train, sensitivity_train, specificity_train, correrate_test, sensitivity_test, specificity_test);
% 
% 


% Support Vector Machine with Linear Kernel

% correrate_train = 0;
% sensitivity_train = 0;
% specificity_train = 0;
% correrate_test = 0;
% sensitivity_test = 0;
% specificity_test = 0;
% for i = 1:10
%     % Training Set
%     X = norm_data(Indices == i,:);
%     Y = new_labels(Indices == i,:);
%     len_data = length(X);
%     train_len = double(len_data - int16(len_data/10));
%     test_len = double(int16(len_data/10));
%     % modeling
%     svm_l = fitcsvm(X(1:train_len,:),Y(1:train_len,:),'KernelFunction','linear');
%     
%     % apply model on training 
%     Y_pred = predict(svm_l,X(1:train_len,:));
%     % compute the confusion matrix
% %     [train_lda_CM,train_grpOrder] = confusionmat(Y(1:(len_data - int16(len_data/10))),class_out_train);
% %     correrate_train = correrate_train + (train_lda_CM(1,1) + train_lda_CM(2,2))/train_len;
%     result_training = classperf(Y(1:train_len,:),Y_pred);
%     correrate_train= correrate_train + result_training.CorrectRate;
%     sensitivity_train = sensitivity_train + result_training.Sensitivity;
%     specificity_train = specificity_train + result_training.Specificity;
%     % 1.4 Apply model on Testing Set
%     Y_pred_test = predict(svm_l,X(train_len+1:end,:));
%     [correrate_test,sensitivity_test, specificity_test] = cfm_test(Y_pred_test,Y,train_len,test_len,correrate_test,sensitivity_test,specificity_test);
% end
% 
% [ann_avg_accy_train,ann_avg_sens_train,ann_avg_spec_train,ann_avg_accy_test,ann_avg_sens_test,ann_avg_spec_test] = avg_eval_matx(correrate_train, sensitivity_train, specificity_train, correrate_test, sensitivity_test, specificity_test);
% 
% 


% Support Vector Machine with Non Linear Kernel

% correrate_train = 0;
% sensitivity_train = 0;
% specificity_train = 0;
% correrate_test = 0;
% sensitivity_test = 0;
% specificity_test = 0;
% for i = 1:10
%     % Training Set
%     X = norm_data(Indices == i,:);
%     Y = new_labels(Indices == i,:);
%     len_data = length(X);
%     train_len = double(len_data - int16(len_data/10));
%     test_len = double(int16(len_data/10));
%     % modeling
%     svm_rbf = fitcsvm(X(1:train_len,:),Y(1:train_len,:),'KernelFunction','rbf');
%     
%     % apply model on training 
%     Y_pred = predict(svm_rbf,X(1:train_len,:));
%     % compute the confusion matrix
% %     [train_lda_CM,train_grpOrder] = confusionmat(Y(1:(len_data - int16(len_data/10))),class_out_train);
% %     correrate_train = correrate_train + (train_lda_CM(1,1) + train_lda_CM(2,2))/train_len;
%     result_training = classperf(Y(1:train_len,:),Y_pred);
%     correrate_train= correrate_train + result_training.CorrectRate;
%     sensitivity_train = sensitivity_train + result_training.Sensitivity;
%     specificity_train = specificity_train + result_training.Specificity;
%     % 1.4 Apply model on Testing Set
%     Y_pred_test = predict(svm_rbf,X(train_len+1:end,:));
%     [test_lda_CM,test_grpOrder] = confusionmat(Y(train_len+1:end),Y_pred_test);
%     [correrate_test,sensitivity_test, specificity_test] = cfm_test(Y_pred_test,Y,train_len,test_len,correrate_test,sensitivity_test,specificity_test);
% end
% 
% [ann_avg_accy_train,ann_avg_sens_train,ann_avg_spec_train,ann_avg_accy_test,ann_avg_sens_test,ann_avg_spec_test] = avg_eval_matx(correrate_train, sensitivity_train, specificity_train, correrate_test, sensitivity_test, specificity_test);


% % Artificial Neural Network(ANN)
% 
% correrate_train = 0;
% sensitivity_train = 0;
% specificity_train = 0;
% correrate_test = 0;
% sensitivity_test = 0;
% specificity_test = 0;
% for i = 1:10
%     % Training Set
%     X = norm_data(Indices == i,:);
%     Y = new_labels(Indices == i,:);
%     len_data = length(X);
%     train_len = double(len_data - int16(len_data/10));
%     test_len = double(int16(len_data/10));
%     % modeling
%     net = newff([0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;],[5 1,1],{'tansig' 'purelin','purelin'});
%     net.trainParam.epochs = 200;
%     net = train(net,X(1:train_len,:)',Y(1:train_len,:)');
%     % apply model on training 
%     Y_pred = sim(net,X(1:train_len,:)');
%     %plot(X',Y',X',Y_pred,'o')
%     Y_pred(Y_pred > 1.5) = 2;
%     Y_pred(Y_pred < 1.5) = 1;
%     % compute the confusion matrix for train set
% %     [train_ann_CM,train_grpOrder] = confusionmat(Y(1:(len_data - int16(len_data/10))),Y_pred);
% %     correrate_train = correrate_train + (train_ann_CM(1,1) + train_ann_CM(2,2))/train_len;
%     result_training = classperf(Y(1:train_len,:),Y_pred');
%     correrate_train= correrate_train + result_training.CorrectRate;
%     sensitivity_train = sensitivity_train + result_training.Sensitivity;
%     specificity_train = specificity_train + result_training.Specificity;
%     
% %     % 1.4 Apply model on Testing Set
%     Y_pred_test = sim(net,X(train_len+1:end,:)');
%     Y_pred_test(Y_pred_test > 1.5) = 2;
%     Y_pred_test(Y_pred_test < 1.5) = 1;
%     
% %     % confusionmat instead classperf since crossvalidation may cause the
% %     % test label to only contain one class
% %     [test_ann_CM,test_grpOrder] = confusionmat(Y(train_len+1:end,:),Y_pred_test);
% %      
% %     if size(test_ann_CM,1) == 1
% %         if(Y(train_len+1,:) == Y_pred_test(1,:))
% %             correrate_test = correrate_test + 1;
% %             sensitivity_test = sensitivity_test + 1;
% %             specificity_test = specificity_test + 1;
% %         else
% %             correrate_test = correrate_test + 0;
% %             sensitivity_test = sensitivity_test + 0;
% %             specificity_test = specificity_test + 0;
% %         end
% %     else
% %         correrate_test = correrate_test + (test_ann_CM(1,1) + test_ann_CM(2,2))/test_len;
% %         sensitivity_test = sensitivity_test + test_ann_CM(1,1)/sum(test_ann_CM(:,1));
% %         specificity_test = specificity_test + test_ann_CM(2,2)/sum(test_ann_CM(:,2));
% %     end
%     
%     [correrate_test,sensitivity_test, specificity_test] = cfm_test(Y_pred_test,Y,train_len,test_len,correrate_test,sensitivity_test,specificity_test);
% end
% 
% [ann_avg_accy_train,ann_avg_sens_train,ann_avg_spec_train,ann_avg_accy_test,ann_avg_sens_test,ann_avg_spec_test] = avg_eval_matx(correrate_train, sensitivity_train, specificity_train, correrate_test, sensitivity_test, specificity_test);

% % Artificial Neural Network(ANN) with non-normalized data
% 
correrate_train = 0;
sensitivity_train = 0;
specificity_train = 0;
correrate_test = 0;
sensitivity_test = 0;
specificity_test = 0;
for i = 1:10
    % Training Set
    X = data(Indices == i,:);
    Y = new_labels(Indices == i,:);
    len_data = length(X);
    train_len = double(len_data - int16(len_data/10));
    test_len = double(int16(len_data/10));
    % modeling
    net = newff([0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;],[5 1,1],{'tansig' 'purelin','purelin'});
    net.trainParam.epochs = 200;
    net = train(net,X(1:train_len,:)',Y(1:train_len,:)');
    % apply model on training 
    Y_pred = sim(net,X(1:train_len,:)');
    %plot(X',Y',X',Y_pred,'o')
    Y_pred(Y_pred > 1.5) = 2;
    Y_pred(Y_pred < 1.5) = 1;
    % compute the confusion matrix for train set
%     [train_ann_CM,train_grpOrder] = confusionmat(Y(1:(len_data - int16(len_data/10))),Y_pred);
%     correrate_train = correrate_train + (train_ann_CM(1,1) + train_ann_CM(2,2))/train_len;
    result_training = classperf(Y(1:train_len,:),Y_pred');
    correrate_train= correrate_train + result_training.CorrectRate;
    sensitivity_train = sensitivity_train + result_training.Sensitivity;
    specificity_train = specificity_train + result_training.Specificity;
    
%     % 1.4 Apply model on Testing Set
    Y_pred_test = sim(net,X(train_len+1:end,:)');
    Y_pred_test(Y_pred_test > 1.5) = 2;
    Y_pred_test(Y_pred_test < 1.5) = 1;
    
%     % confusionmat instead classperf since crossvalidation may cause the
%     % test label to only contain one class

    [correrate_test,sensitivity_test, specificity_test] = cfm_test(Y_pred_test,Y,train_len,test_len,correrate_test,sensitivity_test,specificity_test);
end

[ann_avg_accy_train,ann_avg_sens_train,ann_avg_spec_train,ann_avg_accy_test,ann_avg_sens_test,ann_avg_spec_test] = avg_eval_matx(correrate_train, sensitivity_train, specificity_train, correrate_test, sensitivity_test, specificity_test);
