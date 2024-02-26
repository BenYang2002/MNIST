% Training
labels = loadMNISTLabels('/MATLAB Drive/BackPropagation/train-labels.idx1-ubyte');
labels = labels';
images = loadMNISTImages('/MATLAB Drive/BackPropagation/train-images.idx3-ubyte');
%labels = labels(:,1:10000);
%images = images(:,1:10000);
trainingTimes = 10;
learning_rate = 0.05;
istraining = true;
isMNIST = true;
weightRows = [100,10];
weightColumns = [784,100];
weightRows1 = [1000,100,10];
weightColumns1 = [784,1000,100];
weightRows2 = [1000,100,500,10];
weightColumns2 = [784,1000,100,500];
transferFunc = {"sigmoid","softmax"};
obj_MNIST = BackPropLayer(weightRows,weightColumns,learning_rate,transferFunc,1, ...
    istraining,trainingTimes,isMNIST);
obj_MNIST.train(images,labels);

% Testing
testing_labels = loadMNISTLabels('/MATLAB Drive/BackPropagation/t10k-labels.idx1-ubyte');
testing_labels = testing_labels';
testing_images = loadMNISTImages('/MATLAB Drive/BackPropagation/t10k-images.idx3-ubyte');
%testing_labels = testing_labels(:,1:200);
%testing_images = testing_images(:,1:200);
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
    accuracy(i) = correctCount(i) / totalCount(i);
end
disp(accuracy);

