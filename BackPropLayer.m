classdef BackPropLayer < handle
    %BACKPROPLAYER Summary of this class goes here
    %   Detailed explanation goes here
    properties
        transfer
        %layers contains the argument for each layer
        %eg. layer{1} = {[wMat1,bVect1]}
        layers 
        wMat1
        wMat2
        bVect1
        bVect2
        feedVect
    end
    methods
        function this = BackPropLayer(wMat11,wMat12,wMat21,wMat22,transfer)
        %BACKPROPLAYER Construct an instance of this class
        %   wMati1 means: the first dimension/num of rows of layer i (input
        %   layer is NOT count as a layer)
        %   wMati2 means: the second dimension/num of columns of layer i 
        wMat1 = rand(wMat11,wMat12) * 0.1;
        wMat2 = rand(wMat21,wMat22) * 0.1;
        bVect1 = rand(wMat11,1);
        bVect2 = rand(wMat21,1);
        layers{1} = [wMat1,bVect1];
        layers{2} = [wMat2,bVect2];
        end

        function [output] = forward(this, input)
            %FORWARD
            %Input is a vector from previous layer or from input layer
            this.feedVect = input;
            output = this.wMat * input  + this.bVect;
            output = this.activationFunc(output);
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

        function activationFunc(this,output)
            
        end
        
        function print(this)
            disp("Weight Matrix:");
            disp(this.wMat);
            disp("Bias Vector:");
            disp(this.bVect);
        end
    end
end
