% input for 0
input0 = [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1]';
% input for 1
input1 = [-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]';
% input for 2
input2 = [1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1]';
%output matrix 
output_matrix = [1,0,0;0,1,0;0,0,1];
input_matrix = [input0,input1,input2];

% testing for correctness
obj_sample = BackPropLayer(6,30,3,6,0.01,"sigmoid");
obj_sample.train(input_matrix,output_matrix);
% testing to find out best hidden neuron numbers
