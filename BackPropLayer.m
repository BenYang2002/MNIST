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
    end
    methods
        function this = BackPropLayer(wMat11,wMat12,wMat21,wMat22 ...
                ,learning_rate,transfer)
            %BACKPROPLAYER Construct an instance of this class
            %   wMati1 : the first dimension/num of rows of layer i 
            %   wMati2 : the second dimension/num of columns of layer
            %   i 
            this.wMat1 = rand(wMat11,wMat12) * 0.1;
            this.wMat2 = rand(wMat21,wMat22) * 0.1;
            this.bVect1 = rand(wMat11,1);
            this.bVect2 = rand(wMat21,1);
            this.layers{1} = [this.wMat1,this.bVect1];
            this.layers{2} = [this.wMat2,this.bVect2];
            this.learning_rate = learning_rate;
            this.transfer = transfer;
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
            this.prediction = output;
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
            iter = 0 ;
            correct = false;
            while (iter < 1000 || correct)
                correct = true;
                for i = 1 : size(inputMatrix,2)
                    input = inputMatrix(:,i);
                    ex = expectedM(:,i);
                    this.forward(input);
                    this.backwardUpdate(input,ex);
                    if this.prediction ~= expectedM(:,i)
                        disp("error:");
                        disp("predicion is");
                        disp(this.prediction);
                        disp("expected output is");
                        disp(expectedM(:,i));
                        correct = false;
                    end
                    iter = iter + 1;
                    disp("iter: " + iter);
                end
            end
        end

        function backwardUpdate(this, input,expectedOut)
             %%Compare # of neurons to size of error vector
             % This is the function that updates the weight_matrix based 
             % on a single input
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
