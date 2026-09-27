function [f1Vec, precVec, recVec] = perClassMetrics(cm)
%PERCLASSMETRICS - Computes per-class precision, recall and F1 from a
%                  confusion matrix, where cm(i,j) = true i, predicted j.
%  SYNTAX:
%  [f1Vec, precVec, recVec] = perClassMetrics(cm);
%
%  INPUT ARGUMENTS:
%  - cm  : confusion matrix to compute precision, recall and F1 
%
%  OUTPUT ARGUMENTS:
%  - f1Vec   : vector containing the f1 score for each class
%  - precVec : vector containing the precision value for each class
%  - recVec  : vector containing the recall value for each class

    nClasses = size(cm, 1);
    f1Vec   = zeros(1, nClasses);
    precVec = zeros(1, nClasses);
    recVec  = zeros(1, nClasses);

    for c = 1:nClasses
        TP = cm(c, c);
        FP = sum(cm(:, c)) - TP;
        FN = sum(cm(c, :)) - TP;

        if (TP + FP) > 0
            precVec(c) = TP / (TP + FP);
        end
        if (TP + FN) > 0
            recVec(c) = TP / (TP + FN);
        end
        if (precVec(c) + recVec(c)) > 0
            f1Vec(c) = 2 * precVec(c) * recVec(c) / (precVec(c) + recVec(c));
        end
    end
end