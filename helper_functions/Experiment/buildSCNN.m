function lgraph = buildSCNN(inputSize, numClasses)
%BUILDSCNN - Initialises the layer graph for the standard CNN described in the paper.
%
%  SYNTAX:
%  lgraph = buildSCNN(inputSize, numClasses);
%
%  INPUT ARGUMENTS:
%  - inputSize  : [H W C] size of input images 
%  - numClasses : number of output classes
%
%  OUTPUT ARGUMENTS:
%  - lgraph : layerGraph of the network
%
%  ARCHITECTURE:
%  Standard CNN architecture described in the paper:
%  - 3 convolutional blocks, each with 2x Conv(3x3) + ReLU, then MaxPool(2x2)
%  - Fixed kernel size (3x3)
%  - Filters: 32, 64, 128 
%  - ReLU activation functions
%  - 2 dense blocks with Dropout(0.5)

    layers = [
        imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
        
        % ---- Convolutional Block 1 ----
        convolution2dLayer(3, 32, 'Padding', 'same', ...
            'WeightsInitializer', 'glorot', 'Name', 'conv1_1')
        reluLayer('Name', 'relu1_1')
        convolution2dLayer(3, 32, 'Padding', 'same', ...
            'WeightsInitializer', 'glorot', 'Name', 'conv1_2')
        reluLayer('Name', 'relu1_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        % ---- Convolutional Block 2 ----
        convolution2dLayer(3, 64, 'Padding', 'same', ...
            'WeightsInitializer', 'glorot', 'Name', 'conv2_1')
        reluLayer('Name', 'relu2_1')
        convolution2dLayer(3, 64, 'Padding', 'same', ...
            'WeightsInitializer', 'glorot', 'Name', 'conv2_2')
        reluLayer('Name', 'relu2_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        % ---- Convolutional Block 3 ----
        convolution2dLayer(3, 128, 'Padding', 'same', ...
            'WeightsInitializer', 'glorot', 'Name', 'conv3_1')
        reluLayer('Name', 'relu3_1')
        convolution2dLayer(3, 128, 'Padding', 'same', ...
            'WeightsInitializer', 'glorot', 'Name', 'conv3_2')
        reluLayer('Name', 'relu3_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
        
        % ---- Flatten ----
        flattenLayer('Name', 'flatten')
        
        % ---- FC 1 ----
        fullyConnectedLayer(256, 'WeightsInitializer', 'he', 'Name', 'fc1')
        reluLayer('Name', 'relu_fc1')
        dropoutLayer(0.5, 'Name', 'drop1')
        
        % ---- FC 2 ----
        fullyConnectedLayer(128, 'WeightsInitializer', 'he', 'Name', 'fc2')
        reluLayer('Name', 'relu_fc2')
        dropoutLayer(0.5, 'Name', 'drop2')

        % ---- FC 3 + classification ----
        fullyConnectedLayer(numClasses, 'WeightsInitializer', 'he', 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
    
    lgraph = layerGraph(layers);
end