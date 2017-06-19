% 1.1 Load the data
load('HW1Dataset.mat')

% 1.2 Normalize data
norm_data = normc(data);

% 1.3.1 Cross_validatin_Index
Indices = crossvalind('Kfold', length(norm_data), 10);
% 1.3.2 change label to 1 & 2
new_labels = labels + 1;
% correrate_train = 0;
% correrate_test = 0;
% 
% for i = 1:10
%     % Training Set
%     X = norm_data(Indices == i,:);
%     Y = new_labels(Indices == i,:);
%     len_data = length(X);
%     train_len = double(len_data - int16(len_data/10));
%     test_len = double(int16(len_data/10));
%     % modeling
%     B = mnrfit(X(1:(len_data - int16(len_data/10)),:),Y(1:(len_data - int16(len_data/10)),:));
%     % apply model on training 
%     class_out = mnrval(B,X);
%     % compute the confusion matrix
%     class_out_classified = (class_out(:,1) < class_out(:,2));
%     result_training = classperf(labels(Indices == i),class_out_classified);
%     correrate_train= correrate_train + result_training.CorrectRate;
%     
%     % 1.4 Apply model on Testing Set
%     class_out_test = mnrval(B,X(train_len+1:end,:));
%     class_out_test_classified = (class_out_test(:,1) < class_out_test(:,2));
%     result_test = classperf(train_len+1:end,:),class_out_test_classified);
%     correrate_test = correrate_test + result_test.CorrectRate;
% end
% % Calculate the average accuracy of training and testing set for logistic
% % regression
% lr_avg_cr_train = correrate_train / 10;
% lr_avg_cr_test = correrate_test / 10;
% 
% % Gaussian Naive Bayes 
% correrate_train = 0;
% correrate_test = 0;
% for i = 1:10
%     % Training Set
%     X = norm_data(Indices == i,:);
%     Y = new_labels(Indices == i,:);
%     len_data = length(X);
%     train_len = double(len_data - int16(len_data/10));
%     test_len = double(int16(len_data/10));
%     % modeling
%     B = fitcnb(X(1:(len_data - int16(len_data/10)),:),Y(1:(len_data - int16(len_data/10)),:));
%     % apply model on training 
%     class_out = predict(B,X);
%     % compute the confusion matrix
%     result_training = classperf(labels(Indices == i),class_out);
%     correrate_train= correrate_train + result_training.CorrectRate;
%     
%     % 1.4 Apply model on Testing Set
%     class_out_test = predict(B,X(train_len+1:end,:));
%     result_test = classperf(Y(train_len+1:end,:),class_out_test);
%     correrate_test = correrate_test + result_test.CorrectRate;
% end
% 
% nb_avg_cr_train = correrate_train / 10;
% nb_avg_cr_test = correrate_test / 10;

% Linear Discriminant

correrate_train = 0;
correrate_test = 0;
for i = 1:10
    % Training Set
    X = norm_data(Indices == i,:);
    Y = new_labels(Indices == i,:);
    len_data = length(X);
    train_len = double(len_data - int16(len_data/10));
    test_len = double(int16(len_data/10));
    % modeling
    lda = fitcdiscr(X(1:train_len,:),Y(1:train_len,:));
    
    % apply model on training 
    class_out_train = predict(lda,X(1:train_len,:));
    % compute the confusion matrix
    [train_lda_CM,train_grpOrder] = confusionmat(Y(1:(len_data - int16(len_data/10))),class_out_train);
    correrate_train = correrate_train + (train_lda_CM(1,1) + train_lda_CM(2,2))/train_len;
    
    % 1.4 Apply model on Testing Set
    class_out_test = predict(lda,X(train_len+1:end,:));
    [test_lda_CM,test_grpOrder] = confusionmat(Y(train_len+1:end),class_out_test);
    if size(test_lda_CM,1) == 1
        correrate_test = correrate_test + 1;
    else
        correrate_test = correrate_test + (test_lda_CM(1,1) + test_lda_CM(2,2))/test_len;
    end
end

lda_avg_cr_train = correrate_train / 10;
lda_avg_cr_test = correrate_test / 10;
