clc
clear
% 1.1 Load the data
load('HW1Part2Dataset.mat')

% 1.2 Normalize data
norm_data = normc(data);

% 1.3.1 Cross_validatin_Index
Indices = crossvalind('Kfold', length(norm_data), 10);
% 1.3.2 change label to 1 & 2
new_labels = labels + 1;
ovall_accy = 0; 
ovall_spec = 0;
ovall_sens = 0;
[fs, ovall_accy,ovall_spec,ovall_sens] = hwfs(norm_data,new_labels',Indices);




