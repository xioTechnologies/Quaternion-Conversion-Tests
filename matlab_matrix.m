quaternions = csvread('matlab_quaternions.csv');

csv_rows = zeros(length(quaternions), 13);

for index = 1:length(quaternions)
    csv_rows(index,:) = [quaternions(index, :), reshape(quat2rotm(quaternions(index,:))', 1, [])]; % test: reshape([1 2 3; 4 5 6; 7 8 9]', 1, [])
end

csvwrite('matlab_matrix.csv', csv_rows);
