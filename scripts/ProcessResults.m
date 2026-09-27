%% PROCESS RESULTS
%  Loads all per-run result files and computes aggregated statistics (mean +/- std) 
%  for the performance in the three datatset variants, per-class metrics (precision, 
%  recall, f1), and confusion matrices (mean over all runs for each dataset variant).
%
%  Outputs:
%   - results/overall_results/summary_<exp>.mat                     : aggregated results per dataset
%   - results/overall_results/figures/cm_<exp>.png                  : averaged confusion matrices
%   - results/overall_results/figures/loss_curves_[train/val].png   : plots for the train and test loss
%   - results/overall_results/experiment_results.txt                : table summaries of the experiment (also printed to terminal)

clc; clear; close all;

%% AGGREGATE RESULTS AND COMPUTE MEAN METRICS
% ---- Paths and configuration ----
resultsDir  = fullfile('results');
perRunDir   = fullfile(resultsDir, 'per_run_results');
finDir      = fullfile(resultsDir, 'overall_results');
overallFigDir = fullfile(finDir, 'figures');

if ~exist(finDir, 'dir'), mkdir(finDir); end
if ~exist(overallFigDir, 'dir'), mkdir(overallFigDir); end

% ---- Prepare log file for results tables ----
logFile = fullfile(finDir, 'experiment_results.txt');
logFid  = fopen(logFile, 'w');

experimentNames = {'baseline', 'standardization', 'augmentation'};
nRuns     = 20;      % NOTE: Must be the same as specified in RunExperiments!
nClasses  = 4;

% ---- Results aggregation ----
for e = 1:numel(experimentNames)
    expName = experimentNames{e};

    % Preallocate overall metrics
    accs        = zeros(nRuns, 1);
    macroF1s    = zeros(nRuns, 1);
    finalTrLoss = zeros(nRuns, 1);
    finalVaLoss = zeros(nRuns, 1);
    bestVaLoss  = zeros(nRuns, 1);
    times       = zeros(nRuns, 1);
    epochs      = zeros(nRuns, 1);

    % Preallocate per-class metrics
    perClassF1s   = zeros(nRuns, nClasses);
    perClassPrecs = zeros(nRuns, nClasses);
    perClassRecs  = zeros(nRuns, nClasses);

    % Preallocate confusion matrices (nClasses x nClasses x nRuns)
    confusions = zeros(nClasses, nClasses, nRuns);

    classOrderRef = [];

    % Loop over runs
    for r = 1:nRuns
        f = fullfile(perRunDir, sprintf('result_%s_run%d.mat', expName, r));
        if ~exist(f, 'file')
            error('Missing file: %s', f);
        end
        S = load(f, 'result');
        res = S.result;

        % Aggregated metrics
        accs(r)        = res.testAccuracy;
        macroF1s(r)    = res.testMacroF1;
        finalTrLoss(r) = res.finalTrainLoss;
        finalVaLoss(r) = res.finalValLoss;
        bestVaLoss(r)  = res.bestValLoss;
        times(r)       = res.trainingTime;
        epochs(r)      = res.epochsRun;

        % Per-class metrics
        perClassF1s(r, :)   = res.perClassF1;
        perClassPrecs(r, :) = res.perClassPrec;
        perClassRecs(r, :)  = res.perClassRec;

        % Confusion matrix
        confusions(:, :, r) = res.confusion;

        % Check class order consistency
        if isempty(classOrderRef)
            classOrderRef = res.classOrder;
        elseif ~isequal(classOrderRef, res.classOrder)
            error(['Class order mismatch in %s run %d. ' ...
                   'Expected {%s}, got {%s}.'], ...
                   expName, r, strjoin(classOrderRef, ','), ...
                   strjoin(res.classOrder, ','));
        end
    end

    % ---- Populate summary struct ----
    summary = struct();
    summary.experimentName = expName;
    summary.nRuns          = nRuns;
    summary.classOrder     = classOrderRef;

    % Raw per-run values
    summary.allAccuracies    = accs;
    summary.allMacroF1s      = macroF1s;
    summary.allFinalValLoss  = finalVaLoss;
    summary.allTrainingTimes = times;
    summary.allEpochs        = epochs;
    summary.allPerClassF1    = perClassF1s;
    summary.allPerClassPrec  = perClassPrecs;
    summary.allPerClassRec   = perClassRecs;
    summary.allConfusions    = confusions;

    % Overall means / stds
    summary.meanAccuracy       = mean(accs);
    summary.stdAccuracy        = std(accs);
    summary.meanMacroF1        = mean(macroF1s);
    summary.stdMacroF1         = std(macroF1s);
    summary.meanFinalTrainLoss = mean(finalTrLoss);
    summary.stdFinalTrainLoss  = std(finalTrLoss);
    summary.meanFinalValLoss   = mean(finalVaLoss);
    summary.stdFinalValLoss    = std(finalVaLoss);
    summary.meanBestValLoss    = mean(bestVaLoss);
    summary.stdBestValLoss     = std(bestVaLoss);
    summary.meanTrainingTime   = mean(times);
    summary.stdTrainingTime    = std(times);
    summary.meanEpochsRun      = mean(epochs);
    summary.stdEpochsRun       = std(epochs);

    % Per-class means / stds
    summary.meanPerClassF1   = mean(perClassF1s,   1);
    summary.stdPerClassF1    = std(perClassF1s,    0, 1);
    summary.meanPerClassPrec = mean(perClassPrecs, 1);
    summary.stdPerClassPrec  = std(perClassPrecs,  0, 1);
    summary.meanPerClassRec  = mean(perClassRecs,  1);
    summary.stdPerClassRec   = std(perClassRecs,   0, 1);

    % Confusion matrices: element-wise mean and std
    summary.meanConfusion = mean(confusions, 3);
    summary.stdConfusion  = std(confusions, 0, 3);

    % Save summary
    outFile = fullfile(finDir, sprintf('summary_%s.mat', expName));
    save(outFile, 'summary');
    fprintf('Saved: %s\n', outFile);
end

%% PRINT TABLE (overall scalar metrics)
logPrint(logFid, '\n');
logPrint(logFid, '==========================================================================\n');
logPrint(logFid, '       Average performance of SCNNs (mean +/- std over %d runs)\n', nRuns);
logPrint(logFid, '==========================================================================\n');
logPrint(logFid, ' %-18s | %-15s | %-15s | %-12s\n', ...
    'Pre-processing', 'Macro F1', 'Final Val Loss', 'Time (s)');
logPrint(logFid, '--------------------------------------------------------------------------\n');

for e = 1:numel(experimentNames)
    expName = experimentNames{e};
    S = load(fullfile(finDir, sprintf('summary_%s.mat', expName)));
    s = S.summary;

    logPrint(logFid, ' %-18s | %.3f +/- %.3f | %.3f +/- %.3f | %6.1f +/- %5.1f\n', ...
        s.experimentName, ...
        s.meanMacroF1,    s.stdMacroF1, ...
        s.meanFinalValLoss, s.stdFinalValLoss, ...
        s.meanTrainingTime, s.stdTrainingTime);
end
logPrint(logFid, '==========================================================================\n');

%% PRINT TABLE (per-class metrics)
S = load(fullfile(finDir, sprintf('summary_%s.mat', experimentNames{1})));
classNames = S.summary.classOrder;

logPrint(logFid, '\n');
logPrint(logFid, '=================================================================================================\n');
logPrint(logFid, '             Average per-class performance of SCNNs (mean over %d runs)\n', nRuns);
logPrint(logFid, '=================================================================================================\n');

logPrint(logFid, ' %-16s |', 'Pre-processing');
for c = 1:nClasses
    logPrint(logFid, ' %-17s ', classNames{c});
    if c < nClasses, logPrint(logFid, '|'); end
end
logPrint(logFid, '\n');

logPrint(logFid, ' %-16s |', '');
for c = 1:nClasses
    logPrint(logFid, ' %-16s', '  P     R     F1  ');
    if c < nClasses, logPrint(logFid, '|'); end
end
logPrint(logFid, '\n');
logPrint(logFid, '-------------------------------------------------------------------------------------------------\n');

for e = 1:numel(experimentNames)
    expName = experimentNames{e};
    S = load(fullfile(finDir, sprintf('summary_%s.mat', expName)));
    s = S.summary;

    logPrint(logFid, ' %-16s |', s.experimentName);
    for c = 1:nClasses
        logPrint(logFid, ' %.3f %.3f %.3f ', ...
            s.meanPerClassPrec(c), ...
            s.meanPerClassRec(c), ...
            s.meanPerClassF1(c));
        if c < nClasses, logPrint(logFid, '|'); end
    end
    logPrint(logFid, '\n');
end
logPrint(logFid, '=================================================================================================\n');
fclose(logFid);
fprintf('\nResults tables saved to: %s\n', logFile);

%% GENERATE CONFUSION MATRIX FIGURES
fprintf('\nGenerating confusion matrix figures...\n');

for e = 1:numel(experimentNames)
    expName = experimentNames{e};
    S = load(fullfile(finDir, sprintf('summary_%s.mat', expName)));
    s = S.summary;

    % Row-normalized matrix
    rowSums  = sum(s.meanConfusion, 2);
    normConf = 100 * s.meanConfusion ./ rowSums;

    % Create figure
    fig = figure('Visible', 'off', 'Position', [100 100 700 600]);
    h = heatmap(classNames, classNames, normConf);
    h.CellLabelFormat = '%.2f%%';
    h.Colormap = copper(64);
    h.ColorLimits = [0 100];
    h.FontSize = 11;

    % Labels and title (with Macro F1 and accuracy)
    h.Title = '';
    h.XLabel = 'Predicted label';
    h.YLabel = 'Actual label';

    % Save using print (more robust than exportgraphics with heatmap)
    outPng = fullfile(overallFigDir, sprintf('cm_%s.png', expName));
    print(fig, outPng, '-dpng', '-r150');
    close(fig);
    drawnow;
    pause(0.3);         % Figure generation bug shenanigans

    fprintf('Saved: %s\n', outPng);
end

%% LOGPRINT HELPER
function logPrint(fid, fmt, varargin)
    fprintf(fmt, varargin{:});
    if fid > 0
        fprintf(fid, fmt, varargin{:});
    end
end