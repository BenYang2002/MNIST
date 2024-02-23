%% MNIST

labels = loadMNISTLabels('train-labels.idx1-ubyte');
images = loadMNISTImages('train-images.idx3-ubyte');

testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
testImages = loadMNISTImages('t10k-images.idx3-ubyte');

%disp(testImages(:,1:2));
%disp(testLabels(1:2, 1));
labelsV2 = zeros(size(labels,1),10);

for i = 1:size(labels, 1)
    labelsV2(i, labels(i, 1) + 1) = 1;
end

testLabelsV2 = zeros(size(testLabels,1),10);

for i = 1:size(testLabels, 1)
    testLabelsV2(i, testLabels(i, 1) + 1) = 1;
end
disp(testLabels(1, :));


%% 5 Neurons in hidden layer
figure("Name","5 Neurons");
obj_sample = BackPropLayer([5 10],[size(images, 1) 5],0.01,"sigmoid",0.95,true, 25, false);
obj_sample.train(images(:, 1:1000),labelsV2(1:1000, :)');

%% 10 Neurons in hidden layer
figure("Name","10 Neurons");
obj_sample2 = BackPropLayer([10 10],[size(images, 1) 10],0.01,"sigmoid",0.95,true, 25, false);
obj_sample2.train(images(:, 1:1000),labelsV2(1:1000, :)');

%% 50 Neurons in hidden layer
figure("Name","50 Neurons");
obj_sample3 = BackPropLayer([50 10],[size(images, 1) 50],0.01,"sigmoid",0.95,true, 25, false);
obj_sample3.train(images(:, 1:1000),labelsV2(1:1000, :)');

% disp(obj_sample.forward(testImages(:, 1)));
% disp(obj_sample2.forward(testImages(:, 1)));
% disp(obj_sample3.forward(testImages(:, 1)));

accuracy1 = 0;
accuracy2 = 0;
accuracy3 = 0;

for i = 1:100
    t = testLabelsV2(i, :).';
    a1 = obj_sample.forward(testImages(:, i));
    a2 = obj_sample2.forward(testImages(:, i));
    a3 = obj_sample3.forward(testImages(:, i));

    if (isequal(a1, t))
        accuracy1 = accuracy1 + 1;
    end
    if (isequal(a2, t))
        accuracy2 = accuracy2 + 1;
    end
    if (isequal(a3, t))
        accuracy3 = accuracy3 + 1;
    end
end

disp("Overall accuracy for 5 Neurons is " + accuracy1 + "/100");
disp("Overall accuracy for 10 Neurons is " + accuracy2 + "/100");
disp("Overall accuracy for 50 Neurons is " + accuracy3 + "/100");