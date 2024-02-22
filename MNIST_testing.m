% Training
labels = loadMNISTLabels('/MATLAB Drive/BackPropagation/train-labels.idx1-ubyte');
labels = labels';
images = loadMNISTImages('/MATLAB Drive/BackPropagation/train-images.idx3-ubyte');
trainingTimes = 1;
learning_rate = 0.02;
istraining = true;
isMNIST = true;
weightRows = [1000,10];
weightColumns = [784,1000];
acceptance_rate = 1;
obj_MNIST = BackPropLayer(weightRows,weightColumns,learning_rate,"sigmoid", ...
    acceptance_rate,true,trainingTimes,isMNIST);
obj_MNIST.train(images,labels);

% Testing
testing_labels = loadMNISTLabels('/MATLAB Drive/BackPropagation/t10k-labels.idx1-ubyte');
testing_labels = testing_labels';
obj_MNIST.acceptance_rate = 0.01;
testing_images = loadMNISTImages('/MATLAB Drive/BackPropagation/t10k-images.idx3-ubyte');
obj_MNIST.training = false;
correctCount = zeros(1,10);
totalCount = zeros(1,10);
for i = 1 : size(testing_images,2)
    input = testing_images(:,i);
    ex = testing_labels(:,i);
    obj_MNIST.forward(input);
    totalCount(ex+1) = totalCount(ex+1) + 1;
    if (isequal(ex,obj_MNIST.prediction))
        correctCount(ex+1) = correctCount(ex+1) + 1;
    end
end
accuracy = zeros(1,10);
for i = 1 : size(correctCount,2)
    accuracy = correctCount(i) / totalCount(i);
end
disp(accuracy);
