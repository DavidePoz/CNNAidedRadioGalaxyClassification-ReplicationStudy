function vals_dist = findNumVals(img)
%FINDNUMVALS - Counts the number of unique binned values in a 9x9 neighbourhood of each pixel in the image.
%
%  SYNTAX:
%  vals_dist = findNumVals(img);
%
%  INPUT ARGUMENTS:
%  - img : fits image, normalized in [0, 1]
%
%  OUTPUT ARGUMENTS:
%  - vals_dist : matrix of same size as img, containing for each pixel the
%                number of unique pixel values in its 9x9 neighbourhood

    % Split the interval in 256 bins
    bins = linspace(0, 1, 256);
    
    % count unique binned values in the neighbourhood of each pixel
    countUniqueBins = @(neigh) numel(unique(discretize(neigh, bins)));
    vals_dist = nlfilter(img, [9 9], countUniqueBins);
end