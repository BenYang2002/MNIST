output = [1;1;1];
input = [1;1];
obj = BackPropLayer(3,2,3,3,output,0.1,"sigmoid")
obj.forward(input);
disp(obj.layers{1});
obj.backwardUpdate(input);
disp(obj.layers{1});
