% 1.1 Load the data
load('HW1Dataset.mat')

% 1.2 Normalize data
norm_data = normc(data);

% 1.3.1 Cross_validatin_Index
Indices = crossvalind('Kfold', length(norm_data), 10);
% 1.3.2 change label to 1 & 2
new_labels = labels + 1;
correrate_train = 0;
correrate_test = 0;

for i = 1:10
    % Training Set
    X = norm_data(Indices == i);
    Y = new_labels(Indices == i);
    len_data = length(X);
    % modeling
    B = mnrfit(X(1:(len_data - int16(len_data/10)),:),Y(1:(len_data - int16(len_data/10)),:));
    % apply model on training 
    class_out = mnrval(B,norm_data(Indices == i));
    % compute the confusion matrix
    class_out_classified = (class_out(:,1) < class_out(:,2));
    result_training = classperf(labels(Indices == i),class_out_classified);
    correrate_train= correrate_train + result_training.CorrectRate;
    
    % 1.4 Apply model on Testing Set
    class_out_test = mnrval(B,X(int16(len_data/10)+1:end,:));
    class_out_test_classified = (class_out_test(:,1) < class_out_test(:,2));
    result_test = classperf(Y(int16(len_data/10)+1:end,:),class_out_test_classified);
    correrate_test = correrate_test + result_test.CorrectRate;
end

lr_avg_cr_train = correrate_train / 10;
lr_avg_cr_test = correrate_test / 10;

correrate_train = 0;
correrate_test = 0;
for i = 1:10
    % Training Set
    X = norm_data(Indices == i);
    Y = new_labels(Indices == i);
    len_data = length(X);
    % modeling
    B = fitcnb(X(1:(len_data - int16(len_data/10)),:),Y(1:(len_data - int16(len_data/10)),:));
    % apply model on training 
    class_out = predict(B,norm_data(Indices == i));
    % compute the confusion matrix
    result_training = classperf(labels(Indices == i),class_out);
    correrate_train= correrate_train + result_training.CorrectRate;
    
    % 1.4 Apply model on Testing Set
    class_out_test = predict(B,X(int16(len_data/10)+1:end,:));
    result_test = classperf(Y(int16(len_data/10)+1:end,:),class_out_test);
    correrate_test = correrate_test + result_test.CorrectRate;
end

nb_avg_cr_train = correrate_train / 10;
nb_avg_cr_test = correrate_test / 10;