%%% case study


scores_sum = cell2mat(Dec_values(1,1));
test_sum = cell2mat(test_index(1,1));
scores = scores_sum(:,2);
[sorted_scores, sorted_indices] = sort(scores, 1,'descend'); 
sorted_pairs = test_sum(sorted_indices,:);

sorted_20 = sorted_pairs(1:20,:);
name_p_yanzheng = name_p(:,2:end);

sorted_data = sorted_20; 
name_data = name_p_yanzheng; 


[n_rows_sorted, n_cols_sorted] = size(sorted_data);
[n_rows_name, n_cols_name] = size(name_data);


result_matrix = zeros(n_rows_sorted, n_cols_sorted + 1); 
result_matrix(:, 1:n_cols_sorted) = sorted_data; 

for i = 1:n_rows_sorted
    current_row = sorted_data(i, :);
    
    is_matched = any(all(bsxfun(@eq, name_data, current_row), 2));
    
    result_matrix(i, end) = is_matched;
end

fprintf('总行数: %d\n', n_rows_sorted);
fprintf('匹配成功的行数: %d\n', sum(result_matrix(:, end)));

disp('带匹配标记的结果矩阵:');
disp(result_matrix);



%% Find PMID

circRNA_names = readtable('Association Matrixs.xlsx', 'Sheet', 'CircRNA Names', 'ReadVariableNames', false,'TextType', 'string');
disease_names = readtable('Association Matrixs.xlsx', 'Sheet', 'Disease Names', 'ReadVariableNames', false,'TextType', 'string');

result_table = table((1:size(result_matrix, 1))', ...
                    strings(size(result_matrix, 1), 1), ...
                    strings(size(result_matrix, 1), 1), ...
                    'VariableNames', {'No', 'CircRNA_Name', 'Disease_Name'});

for i = 1:size(result_matrix, 1)
    circRNA_idx = result_matrix(i, 1);
    disease_idx = result_matrix(i, 2);
    
    if circRNA_idx > 0 && circRNA_idx <= height(circRNA_names)
        result_table.CircRNA_Name(i) = string(circRNA_names{circRNA_idx, 1}{1});
    else
        result_table.CircRNA_Name(i) = "未匹配";
    end
    
    if disease_idx > 0 && disease_idx <= height(disease_names)
        result_table.Disease_Name(i) = string(disease_names{disease_idx, 1}{1});
    else
        result_table.Disease_Name(i) = "未匹配";
    end
end

output_filename = 'CircRNA_Disease_Pairs.xlsx';
writetable(result_table, output_filename, 'Sheet', 'Results');

fprintf('结果已保存到 %s\n', output_filename);

%% FIND PMID
circR2DiseaseData = readtable('CircR2Disease.xlsx', 'Sheet', 'Sheet1', 'TextType', 'string');
circPairsData = readtable('CircRNA_Disease_Pairs.xlsx', 'TextType', 'string');

circR2DiseaseData.circRNA_Name = erase(circR2DiseaseData{:, 1}, "'");
circPairsData.CircRNA_Name = erase(circPairsData{:, 2}, "'");

pmid = strings(height(circPairsData), 1);

for i = 1:height(circPairsData)
    circRNA = circPairsData.CircRNA_Name(i); 
    disease = circPairsData{i, 3};     
    
    matchIdx = find(strcmpi(circR2DiseaseData.circRNA_Name, circRNA) & ...
                   strcmpi(circR2DiseaseData{:, 6}, disease));
    
    if ~isempty(matchIdx)
        pmid(i) = string(circR2DiseaseData{matchIdx(1), 12}); 
    else
        pmid(i) = "Not Found"; 
    end
end

circPairsData.PMID = pmid;

writetable(circPairsData, 'Updated_CircRNA_Disease_Pairs.xlsx');
