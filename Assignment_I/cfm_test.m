function [correrate_test,sensitivity_test,specificity_test] = cfm_test(Y_pred_test,Y,train_len,test_len,correrate_test,sensitivity_test,specificity_test)
   
    
    % confusionmat instead classperf since crossvalidation may cause the
    % test label to only contain one class
    [test_CM,test_grpOrder] = confusionmat(Y(train_len+1:end,:),Y_pred_test);
     
    if size(test_CM,1) == 1
        if(Y(train_len+1,:) == Y_pred_test(1,:))
            correrate_test = correrate_test + 1;
            sensitivity_test = sensitivity_test + 1;
            specificity_test = specificity_test + 1;
        else
            correrate_test = correrate_test + 0;
            sensitivity_test = sensitivity_test + 0;
            specificity_test = specificity_test + 0;
        end
    else
        try
            result_training = classperf(Y(1:train_len,:),Y_pred);
            correrate_test = correrate_test + result_training.CorrectRate;
            sensitivity_test = sensitivity_test + result_training.Sensitivity;
            specificity_test = specificity_test + result_training.Specificity;
        catch ME
            correrate_test = correrate_test + (test_CM(1,1) + test_CM(2,2))/test_len;
            sensitivity_test = sensitivity_test + test_CM(1,1)/sum(test_CM(:,1));
            specificity_test = specificity_test + test_CM(2,2)/sum(test_CM(:,2));
        end
    end

    