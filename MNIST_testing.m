% Training
labels = loadMNISTLabels('/MATLAB Drive/BackPropagation/train-labels.idx1-ubyte');
labels = labels';
images = loadMNISTImages('/MATLAB Drive/BackPropagation/train-images.idx3-ubyte');
trainingTimes = 1;
learning_rate = 0.01;
istraining = true;
isMNIST = true;
obj_MNIST = BackPropLayer(100,784,10,100,learning_rate,"sigmoid",1,true, ...
    trainingTimes,isMNIST);
obj_MNIST.train(images,labels);

% Testing
testing_labels = loadMNISTLabels('/MATLAB Drive/BackPropagation/t10k-labels.idx1-ubyte');
testing_labels = testing_labels';
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
