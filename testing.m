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
obj_sample = BackPropLayer(6,30,3,6,0.01,"sigmoid",0.95,true);
obj_sample.train(input_matrix,output_matrix);

flip = 2;
obj_sample.training = false;
accuracy1 = [];
accuracy2 = [];
accuracy3 = [];
for j = 0 : flip
    % this vector keep track of the correct times for three kinds of input
    number_flipped = 4 * j;
    correct_times = [0,0,0];
    flip_input = zeros(30,3);
    prediction = zeros(3,3);
    test_size = 1000;
    for i = 1 : test_size
        for k = 1:3
            filp_input(:,k) = addNoise(input_matrix(:,k),number_flipped);
            prediction(:,k) = obj_sample.forward(filp_input(:,k));
            if isequal(prediction(:,k),output_matrix(:,k))
                correct_times(k) = correct_times(k) + 1;
            end
        end
    end
    disp("correct rate for flip " + number_flipped + " pixels for pattern 0 is: " + correct_times(1)/test_size);
    disp("correct rate for flip " + number_flipped + " pixels for pattern 1 is: " + correct_times(2)/test_size);
    disp("correct rate for flip " + number_flipped + " pixels for pattern 2 is: " + correct_times(3)/test_size);
    disp(" ");
    accuracy1 = [accuracy1,correct_times(1)/test_size];
    accuracy2 = [accuracy2,correct_times(2)/test_size];
    accuracy3 = [accuracy3,correct_times(3)/test_size];
end
% testing to find out best hidden neuron numbers
