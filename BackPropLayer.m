classdef BackPropLayer < handle
    %BACKPROPLAYER Summary of this class goes here
    %   Detailed explanation goes here
    properties
        transfer
        layers          % layers is a cell array
        % layers contains the argument for each layer
        %eg. layer{1} = {[wMat1,bVect1]}
        wMat1
        wMat2
        bVect1
        bVect2

        aLayers         % aLayers is a cell array
        % aLayers{i} contains the output of layer i + 1
        % eg aLayer{i} = a(i+1) and the input is counted as the first
        % output

        nLayers         % nlayers is a cell arrya
        % nLayers{i} contains the netinput of layer i

        prediction

        feedVect
        sensitivity_Matrix % sensitivity_Matrix is a numeric matrix
        % each column is the corresponding layer's sensitivity

        learning_rate

        acceptance_rate % after reaching the rate, we will cast the 
        % output vector and make a prediction
        % eg. if ar = 0.95, only an output element has a value larger than
        % 0.95 that we say the element is the correct output
        % [0.2,0.2,0.6] = [0.2,0.2,0.6] we don't make prediction
        % [0.01,0.01,0.98] = [0,0,1] we predict it's 2

        training      % this boolean indicates whether it is training or in 
        % predicion mode. In prediction mode, solution will always be
        % casted
        
        trainingTimes % number of max times going through the training set

        MNIST         % boolean that is used to modify output from a vector
        % to a real number. eg. [0,0,1,0,0,0,0,0,0,0] to 3;

    end
    methods
        function this = BackPropLayer(weightRow, weightColumn, ...
                learning_rate,transfer,acceptance, training, ...
                trainingTimes,MNIST)
            %BACKPROPLAYER Construct an instance of this class
            if (size(weightRow,2) ~= size(weightColumn,2))
                error("dimention of weightRow and weightColumn " + ...
                    "doesn't match");
            end
            for i = 1 : size(weightColumn,2)
                weightMatrix = rand(weightRow(i),weightColumn(i)) * 0.1;
                biasVec = rand(weightRow(i),1);
                this.layers{i} = [weightMatrix,biasVec];
            end
            this.learning_rate = learning_rate;
            this.transfer = transfer;
            this.acceptance_rate = acceptance;
            this.training = training;
            this.trainingTimes = trainingTimes;
            this.MNIST = MNIST;
        end


        function [output] = forward(this, input)
            %FORWARD
            %Input is a vector from previous layer or from input layer
            this.feedVect = input;
            this.aLayers{1} = input;
            for i = 1 : size(this.layers,2)
                parameterM = this.layers{i};
                % this.aLayers{i} is the ath input parameter
                % from first to end - 1 columns are the weight matrix
                % with the last column as bias 
                layerNetInput = parameterM(:,1:end-1) * this.feedVect + ...
                    parameterM(:,end); % net input
                this.nLayers{i} = layerNetInput;
                layerOut = this.activationFunc(layerNetInput,this.transfer);
                this.feedVect = layerOut;
                this.aLayers{i+1} = layerOut;
            end
            output = this.aLayers{end};
            output = this.modifyOutput(output);
            this.prediction = output;
        end

        function output = modifyOutput(this,input)
            max = 0;
            out = 0;
            for i = 1:size(input,1)
                if max < input(i)
                    max = input(i);
                    out = i;
                end
            end
            if (max > this.acceptance_rate || ~this.training)
                if (this.MNIST)
                    output = out - 1;
                    return;
                end
                output = zeros(size(input,1),1);
                output(out) = 1;
                return;
            end
            output = input;
        end

        function der = takeDeravative(this,funcName,input)
            % funcName specify the activation function name
            % Input is the netInput of m layer
            if (funcName == "sigmoid")
                result = this.sigmoid(input);
                der = result' * (1 - result);
            end
        end

        function train(this, inputMatrix, expectedM)
            % sample training 
            % TODO : Decide whether to update the terminating condition
            % inputMatrix assume that each column is an input
            % expectedM each column is one corresponding expected output
            epoch = 1 ;
            correct = false;
            while (~correct && epoch <= this.trainingTimes)
                correct = true;                   
                disp("iter: " + epoch);
                iter = 1;
                for i = 1 : size(inputMatrix,2)
                    disp("iter: " + iter);
                    iter = iter + 1;
                    input = inputMatrix(:,i);
                    ex = expectedM(:,i);
                    this.forward(input);
                    this.backwardUpdate(input,ex);
                    if ~isequal(this.prediction,expectedM(:,i))
                        %disp("error:");
                        %disp("predicion is");
                        %disp(this.prediction);
                        %disp("expected output is");
                        %disp(expectedM(:,i));
                        correct = false;
                    end
                end
                ex = expectedM(:, size(inputMatrix, 2));
                pIndex = (ex - this.prediction) * (ex - this.prediction).';
                plot(zeros(length(this.prediction), 1) + epoch, pIndex);
                if mod(epoch, 20) == 0
                    drawnow();
                end
                hold on
                epoch = epoch + 1;
            end
        end

        function backwardUpdate(this, input,expectedOut)
             %%Compare # of neurons to size of error vector
             % This is the function that updates the weight_matrix based 
             % on a single input
             if (this.MNIST)
                 exOutMod = zeros(10,1);
                 exOutMod(expectedOut+1) = 1; 
                 expectedOut = exOutMod; % we map the output from a scalar 
                 % to the vector
             end
             errorOut = expectedOut - this.prediction;
             der = this.takeDeravative(this.transfer,input);
             sM = -2 * der * (errorOut); % calculated the sensitivity for
             % the last layer
             this.sensitivity_Matrix{size(this.layers,2)} = [sM];
             prevSense = this.sensitivity_Matrix{end};
             % calculate all sensitivity
             for i = size(this.layers,2) : 2
                netV = cell2mat(this.nLayers(:,i));
                der = this.takeDeravative(this.transfer,netV);
                sCurrent = der * this.layers{i}(:,1:end-1)' * prevSense;
                % sCurrent is the sensitivity of the current layer
                prevSense = sCurrent; 
                this.sensitivity_Matrix{i-1} = ...
                    sCurrent;
             end
             % now we have the sensitivity matrix 
             % update weight matrix and bias
             for i = 1 : size(this.layers,2) 
                wM = this.layers{i}(:,1:end - 1); % weight matrix
                b = this.layers{i}(:,end); % bias
                s = this.sensitivity_Matrix{i};
                prevA = (this.aLayers{i})';
                wM = wM - this.learning_rate * s * prevA;
                b = b - this.learning_rate * s;
                this.layers{i} = [wM,b];
             end
        end

        function output = activationFunc(this,input,funcName)
           if (funcName == "sigmoid")
               output = this.sigmoid(input);
               return;
           end
           output = input;
        end

        function output = sigmoid(this,input)
            output = input;
            for i = 1 : size(input,1)
                output(i) = 1 / (1 + exp(1)^(-input(i)));
            end
        end
        
        function print(this)
            for i = 1 : size(this.layers,2)     
                disp("Weight Matrix for layer " + i);
                disp(this.layers{i}(:,1:end-1));
                disp("Bias Vector for layer " + i);
                disp(this.layers{i}(:,end));
            end
        end
    end
end
