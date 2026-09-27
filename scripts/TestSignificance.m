%% STATISTICAL TESTS ON EXPERIMENT RESULTS
%  Welch t-tests with Bonferroni correction.
%  For each pair of preprocessing strategies, this script tests whether their
%  macro F1 distributions have significantly different means.
%  Welch's t-test is used (does not assume equal variances).
%  Bonferroni correction is applied for 3 simultaneous comparisons:
%      alpha_corrected = 0.05 / 3 = 0.0167
%
%  Results are printed to the terminal and saved to results/overall_results/significance_result.txt

clc; clear; close all;

% ---- Paths ----
finDir = fullfile('results', 'overall_results');
logFile = fullfile(finDir, 'significance_result.txt');
logFid  = fopen(logFile, 'w');

experimentNames = {'baseline', 'standardization', 'augmentation'};
nComparisons    = 3;
alphaRaw        = 0.05;
alphaCorrected  = alphaRaw / nComparisons;

% ---- Load macro F1 per-run values ----
macroF1 = cell(numel(experimentNames), 1);
for e = 1:numel(experimentNames)
    S = load(fullfile(finDir, sprintf('summary_%s.mat', experimentNames{e})));
    macroF1{e} = S.summary.allMacroF1s(:);
end

% ---- Pairs to compare ----
pairs = { ...
    1, 2;   % baseline        vs standardization
    1, 3;   % baseline        vs augmentation
    2, 3    % standardization vs augmentation
};

pairLabels = { ...
    'Baseline vs Standardization', ...
    'Baseline vs Augmentation', ...
    'Standardization vs Augmentation'
};

% ---- Run Welch t-tests ----
logPrint(logFid,'\n');
logPrint(logFid,'=================================================================================\n');
logPrint(logFid,' Welch two-sample t-tests\n');
logPrint(logFid,' Bonferroni-corrected alpha = %.4f (%d comparisons)\n', ...
        alphaCorrected, nComparisons);
logPrint(logFid,'=================================================================================\n');
logPrint(logFid,' %-32s | %-9s | %-10s | %-8s | %s\n', ...
    'Comparison', 't-stat', 'p-value', 'Sig.?', 'Decision');
logPrint(logFid,'---------------------------------------------------------------------------------\n');

results = struct('pair', {}, 'tstat', {}, 'pvalue', {}, 'significant', {});

for k = 1:size(pairs, 1)
    a = pairs{k, 1};
    b = pairs{k, 2};

    x = macroF1{a};
    y = macroF1{b};

    % Welch's two-sample t-test
    [h, p, ci, stats] = ttest2(x, y, 'Vartype', 'unequal', ...
                               'Alpha', alphaCorrected);

    sigStr = 'NO';
    if h, sigStr = 'YES'; end

    logPrint(logFid,' %-32s | %+9.3f | %10.2e | %-8s | ', ...
        pairLabels{k}, stats.tstat, p, sigStr);

    if h
        logPrint(logFid,'Reject H0\n');
    else
        logPrint(logFid,'Fail to reject H0\n');
    end

    results(k).pair        = pairLabels{k};
    results(k).tstat       = stats.tstat;
    results(k).df          = stats.df;
    results(k).pvalue      = p;
    results(k).ci          = ci;
    results(k).significant = h;
end

logPrint(logFid,'=================================================================================\n');

fclose(logFid);
fprintf('\nResults tables saved to: %s\n', logFile);

%% LOGPRINT HELPER
function logPrint(fid, fmt, varargin)
    fprintf(fmt, varargin{:});
    if fid > 0
        fprintf(fid, fmt, varargin{:});
    end
end