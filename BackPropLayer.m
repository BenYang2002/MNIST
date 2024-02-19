classdef BackPropLayer < handle
    %BACKPROPLAYER Summary of this class goes here
    %   Detailed explanation goes here
    properties
        transfer
        layers          %layers contains the argument for each layer
        %eg. layer{1} = {[wMat1,bVect1]}
        wMat1
        wMat2
        bVect1
        bVect2
        aLayers         %aLayers{i} contains the output of layer i
        % eg aLayer{i} = ai
        feedVect
    end
    methods
        function this = BackPropLayer(wMat11,wMat12,wMat21,wMat22,transfer)
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
            this.transfer = transfer;
        end

        function [output] = forward(this, input)
            %FORWARD
            %Input is a vector from previous layer or from input layer
            this.feedVect = input;
            this.aLayers{1} = input;
            for i = 1 : size(this.layers,2)
                parameterM = this.layers{i};
                disp(parameterM);
                % this.aLayers{i} is the ath input parameter
                % from first to end - 1 columns are the weight matrix
                % with the last column as bias 
                layerNetInput = parameterM(:,1:end-1) * this.feedVect
                + parameterM(:,end); % net input
                layerOut = this.activationFunc(layerNetInput);
                this.feedVect = layerOut;
                this.aLayers{i+1} = layerOut;
            end
            output = this.aLayers{end};
        end

        function backwardUpdate(this, error)
             %%Compare # of neurons to size of error vector
             if size(this.wMat, 1) == size(error, 1)
                this.wMat = this.wMat + error.' * this.feedVect;
                this.bVect = this.bVect + error;
             else
                error("Error: Neuron size and error size don't match.");
             end
        end

        function output = activationFunc(this,input)
           output = input;
        end
        
        function print(this)
            disp("Weight Matrix:");
            disp(this.wMat);
            disp("Bias Vector:");
            disp(this.bVect);
        end
    end
end
