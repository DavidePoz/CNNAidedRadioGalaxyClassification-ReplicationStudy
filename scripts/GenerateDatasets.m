%% GENERATE PROCESSED DATASETS
%  Creates three datastore variants:
%   - Baseline:         original images (already in datastores.mat)
%   - Standardized:     stdRotation applied to all splits
%   - Augmented:        6 rotations (60 deg intervals), only on training images

clc; close all;

% Paths
dataDir = fullfile('data');
procDir = fullfile(dataDir, 'processed');

if ~exist(procDir, 'dir'), mkdir(procDir); end

% Load original datastores
load(fullfile(dataDir, 'datastores.mat'));
fprintf('Loaded originals: Train=%d, Val=%d, Test=%d\n', ...
    numel(imdsTrain.Files), numel(imdsVal.Files), numel(imdsTest.Files));

% Generate standardized dataset
fprintf('\n=== Standardization ===\n');
stdDir = fullfile(procDir, 'standardized');
if exist(stdDir, 'dir'), rmdir(stdDir, 's'); end
mkdir(stdDir);

imdsTrainStd = standardizeDatastore(imdsTrain, fullfile(stdDir, 'train'));
imdsValStd   = standardizeDatastore(imdsVal,   fullfile(stdDir, 'val'));
imdsTestStd  = standardizeDatastore(imdsTest,  fullfile(stdDir, 'test'));

% Generate augmented dataset (Training only)
fprintf('\n=== Augmentation (training only) ===\n');
augDir = fullfile(procDir, 'augmented');
if exist(augDir, 'dir'), rmdir(augDir, 's'); end
mkdir(augDir);

imdsTrainAug = augmentDatastore(imdsTrain, 0:60:300, augDir);

% Save datastores
outFile = fullfile(dataDir, 'datastores_processed.mat');
save(outFile, ...
    'imdsTrain', 'imdsVal', 'imdsTest', ...
    'imdsTrainStd', 'imdsValStd', 'imdsTestStd', ...
    'imdsTrainAug');

fprintf('\nDatastores saved to %s\n\n', outFile);
fprintf('  Baseline train:   %d\n', numel(imdsTrain.Files));
fprintf('  Baseline val:   %d\n', numel(imdsVal.Files));
fprintf('  Baseline test:   %d\n\n', numel(imdsTest.Files));
fprintf('  Standardized train: %d\n', numel(imdsTrainStd.Files));
fprintf('  Standardized val: %d\n', numel(imdsValStd.Files));
fprintf('  Standardized test: %d\n\n', numel(imdsTestStd.Files));
fprintf('  Augmented train:  %d\n', numel(imdsTrainAug.Files));