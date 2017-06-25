function [ann_avg_accy_train,ann_avg_sens_train,ann_avg_spec_train,ann_avg_accy_test,ann_avg_sens_test,ann_avg_spec_test] = avg_eval_matx(correrate_train, sensitivity_train, specificity_train, correrate_test, sensitivity_test, specificity_test)
    ann_avg_accy_train = correrate_train / 10;
    ann_avg_sens_train = sensitivity_train / 10;
    ann_avg_spec_train = specificity_train / 10;
    ann_avg_accy_test = correrate_test / 10;
    ann_avg_sens_test = sensitivity_test / 10;
    ann_avg_spec_test = specificity_test / 10;
end