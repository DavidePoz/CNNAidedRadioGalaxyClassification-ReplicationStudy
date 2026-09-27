%% RUN EXPERIMENTS: Baseline, Standardization, Augmentation
%  During an experiment, the SCNN is trained multiple times on the three dataset configurations.
%  Each trained model is evaluated on its appropriate test set.
%
%  An experiment consists of nRuns independent runs on each dataset
%  configuration, with each result saved to [results/per_run/result_<exp>_run<N>.mat] 
%
%  Optionally, models are saved to [results/models/model_<exp>_run<N>.mat].
%  NOTE: To save the trained models, uncomment lines 152-153

clc; close all;

% ---- Paths ----
dataDir    = fullfile('data');                                      % Directory with the dataset
resultsDir = fullfile('results');                                   % Directory to store results
perRunResultsDir  = fullfile(resultsDir, 'per_run_results');        % Directory to store results of individual runs
modelsDir  = fullfile(resultsDir, 'models');                        % Directory to store resulting models

for d = {resultsDir, perRunResultsDir, modelsDir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

% ---- Load datastores ----
load(fullfile(dataDir, 'datastores_processed.mat'));
fprintf('Datastores loaded.\n');

% ---- Define experiments ----
experiments = struct( ...
    'name',    {'baseline', 'standardization', 'augmentation'}, ...
    'trainDS', {imdsTrain, imdsTrainStd, imdsTrainAug}, ...
    'valDS',   {imdsVal,   imdsValStd,   imdsVal}, ...
    'testDS',  {imdsTest,  imdsTestStd,  imdsTest} ...
);

% ---- Experiment configuration ----
inputSize  = [150 150 1];       % Image input size
numClasses = 4;                 % Output classes
nRuns      = 20;                % Runs per sub-experiment (baseline, standardized, augmented)
miniBatch  = 32;                % Mini-batch size
valFreq    = 10;                % Validation check every valFreq iterations
epochPatience = 5;              % Stop training if val loss doesn't decrease for epochPatience epochs
maxEpochs  = 100;               % Max training epochs
learnRate  = 1e-4;              % Learning rate

fprintf('\n=====================================================\n');
fprintf(' Starting experiments: %d runs x %d configs\n', nRuns, numel(experiments));
fprintf('=====================================================\n');

for e = 1:numel(experiments)
    % Exp = {baseline, standardized, augmented}
    exp = experiments(e);

    for run = 1:nRuns
        fprintf('\n--- Experiment: %s | Run %d / %d ---\n', exp.name, run, nRuns);

        % ---- Init new network ----
        lgraph = buildSCNN(inputSize, numClasses);

        % ---- Validation patience ----
        nTrainImgs    = numel(exp.trainDS.Files);
        itersPerEpoch = floor(nTrainImgs / miniBatch);
        valPatience   = ceil(epochPatience * itersPerEpoch / valFreq);

        % ---- Training options ----
        options = trainingOptions('adam', ...
            'InitialLearnRate',  learnRate, ...
            'MaxEpochs',         maxEpochs, ...
            'MiniBatchSize',     miniBatch, ...
            'Shuffle',           'every-epoch', ...
            'ValidationData',    exp.valDS, ...
            'ValidationFrequency', valFreq, ...
            'ValidationPatience', valPatience, ...
            'Verbose',           true, ...
            'VerboseFrequency',  20, ...
            'Plots',             'none', ...
            'ExecutionEnvironment', 'auto');

        % ---- Train ----
        tic;
        [net, info] = trainNetwork(exp.trainDS, lgraph, options);
        elapsed = toc;

        % ---- Evaluate on test set ----
        [preds, ~] = classify(net, exp.testDS);
        trueLabels = exp.testDS.Labels;
        classOrder = categories(trueLabels);
        accuracy = mean(preds == trueLabels);
        cm       = confusionmat(trueLabels, preds, 'Order', classOrder);

        % ---- Per-class metrics ----
        [perClassF1, perClassPrec, perClassRec] = perClassMetrics(cm);
        macroF1 = mean(perClassF1);

        fprintf('Test accuracy: %.4f | Macro F1: %.4f | Time: %.1f s\n', ...
            accuracy, macroF1, elapsed);

        % ---- Overfitting indicators ----
        finalTrainLoss = info.TrainingLoss(end);
        finalValLoss   = info.ValidationLoss(end);
        epochsRun     = numel(info.TrainingLoss) / itersPerEpoch;
        nValChecks    = numel(info.ValidationLoss);
        valIterations = valFreq * (1:nValChecks);
        [bestValLoss, bestIdx] = min(info.ValidationLoss);
        bestIteration = valIterations(bestIdx);
        bestEpoch     = bestIteration / itersPerEpoch;

        % ---- Populate the result struct ----
        result = struct();

        % Identification
        result.experimentName = exp.name;
        result.runNumber      = run;
        result.timestamp      = datestr(now);

        % Performance on test set
        result.testAccuracy = accuracy;
        result.testMacroF1  = macroF1;
        result.classOrder   = classOrder;
        result.perClassF1   = perClassF1;
        result.perClassPrec = perClassPrec;
        result.perClassRec  = perClassRec;
        result.confusion    = cm;
        result.predictions  = preds;
        result.trueLabels   = trueLabels;

        % Efficiency
        result.trainingTime  = elapsed;
        result.epochsRun     = epochsRun;
        result.itersPerEpoch = itersPerEpoch;
        result.bestEpoch     = bestEpoch;
        result.bestIteration = bestIteration;

        % Overfitting indicators
        result.finalTrainLoss = finalTrainLoss;
        result.finalValLoss   = finalValLoss;
        result.bestValLoss    = bestValLoss;

        % Loss curves
        result.lossCurveTrain = info.TrainingLoss;
        result.lossCurveVal   = info.ValidationLoss;

        % Accuracy curves
        result.accCurveTrain = info.TrainingAccuracy;
        result.accCurveVal   = info.ValidationAccuracy;

        % ---- Save result ----
        resultFile = fullfile(perRunResultsDir, ...
            sprintf('result_%s_run%d.mat', exp.name, run));
        save(resultFile, 'result', '-v7.3');

        % ---- UNCOMMENT to save trained model ----
        %modelFile = fullfile(modelsDir, sprintf('model_%s_run%d.mat', exp.name, run));
        %save(modelFile, 'net', '-v7.3');
    end
end

fprintf('\n=====================================================\n');
fprintf(' All experiments completed.\n');
fprintf('=====================================================\n');