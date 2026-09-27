%% LOAD DATASET AND SPLIT INTO TRAIN, VALIDATION AND TEST
%  This script loads the images and their labels from the dataset,
%  splits them into training, validation and testing, and
%  saves the resulting datastores.

clc; close all;

dataDir   = fullfile('data', 'FRGMRC-221022-RELEASE-V1.0');
labelFile = fullfile(dataDir, 'FRGMRC-221022-RELEASE-V1.0.csv');

% Extract labels from csv
T = readtable(labelFile, 'VariableNamingRule', 'preserve');
T.Class = extractBefore(T.NAME, '_');
labels = categorical(T.Class);
fprintf('- Labels extracted\n');

% Dataset file paths
fullPaths = fullfile(dataDir, strcat(T.NAME, '.fits'));
fprintf('- Paths etracted\n');

% Create datastore
imds = imageDatastore(fullPaths, ...
    'Labels', labels, ...
    'ReadFcn', @readFitsImage);
fprintf('- Datastore created\n');

% Split into training, validation and test 
trainRatio = 0.80;
valRatio = 0.10;

[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, trainRatio, valRatio, 'randomized');
fprintf('- Dataset split\n');

% Save datastores
save(fullfile('data', 'datastores.mat'), 'imdsTrain', 'imdsVal', 'imdsTest');
fprintf('- Datastores saved.\n  Training: %d | Validation: %d | Test: %d\n', ...
    length(imdsTrain.Files), length(imdsVal.Files), length(imdsTest.Files));