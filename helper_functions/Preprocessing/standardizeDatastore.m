function imdsOut = standardizeDatastore(imds, outDir)
%STANDARDIZEDATASTORE - Applies rotational standardization to each image in imds
%                       and saves the result to outDir
%
%  SYNTAX:
%  imdsOut = standardizeDatastore(imds, outDir);
%
%  INPUT ARGUMENTS:
%  - imds   : the datastore whose images are to be standardized
%  - outDir : destintation directory to save the resulting images
%
%  OUTPUT ARGUMENTS:
%  - imdsOut : standardized version of imds
    
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    n = numel(imds.Files);
    newFiles = cell(n, 1);
    
    % Generate standardized images
    for i = 1:n
        
        % Read image, apply thresholding and rotational standardization
        img = readimage(imds, i);
        mask    = thresholdImage(img);
        img_std = stdRotation(mask, img);
        
        % Resulting file
        fname = fullfile(outDir, sprintf('std_%04d.mat', i));
        save(fname, 'img_std');
        newFiles{i} = fname;
        
        % Print progress
        if mod(i, 50) == 0 || i == n
            fprintf('  [%s] %d / %d\n', outDir, i, n);
        end
    end
    
    % Generate resulting datastore
    imdsOut = imageDatastore(newFiles, 'Labels', imds.Labels, ...
    'FileExtensions', '.mat', ...
    'ReadFcn', @(f) load(f, 'img_std').img_std);
end