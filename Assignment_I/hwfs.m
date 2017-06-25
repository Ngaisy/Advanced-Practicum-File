function [sf, ovall_accy,ovall_spec,ovall_sens] = hwfs(norm_data,new_labels,Indices)
    
    correrate_test = 0;
    sensitivity_test = 0;
    specificity_test = 0;
    ovall_accy = 0; 
    ovall_spec = 0;
    ovall_sens = 0;
    
    prev_corr_test = 0.1;
    notsf = 1:57;
    sf = [];
    diff = 0.02;
    num_diff_sm001_times = 0;
    while diff > 0.01 && num_diff_sm001_times < 8 
        % loop through all the element in not-selected feature
        for j = notsf(:)'
            % 10 fold
            for i = 1:10
                tst_f = [j,sf];
              % Training Set
                X = norm_data(Indices == i,:);
                Y = new_labels(Indices == i,:);
                len_data = length(X);
                train_len = double(len_data - int16(len_data/10));
                test_len = double(int16(len_data/10));
                % modeling
                lda = fitcdiscr(X(1:train_len,tst_f),Y(1:train_len,:));
                % 1.4 Apply model on Testing Set
                Y_pred_test = predict(lda,X(train_len+1:end,tst_f));
                [correrate_test,sensitivity_test, specificity_test] = cfm_test(Y_pred_test,Y,train_len,test_len,correrate_test,sensitivity_test,specificity_test);
            end

            if prev_corr_test < correrate_test
                tmp_sf = j;
                % Calculate the difference between previous and current test
                diff = (correrate_test - prev_corr_test)/prev_corr_test;
                if diff > 0.01
                    num_diff_sm001_times = num_diff_sm001_times + 1;
                end
                prev_corr_test = correrate_test;
                correrate_test = 0;
            end
            sf = [sf,tmp_sf];
            notsf(tmp_sf) = [];
        end
        ovall_accy = 0; 
        ovall_spec = 0;
        ovall_sens = 0;
    end
end