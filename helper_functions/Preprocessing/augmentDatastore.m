function imdsOut = augmentDatastore(imds, angles, outDir)
%AUGMENTDATASTORE - Creates rotated copies of each image in imds and saves
%                   them to the specified output directory.
%
%  SYNTAX:
%  imdsOut = augmentImage(imds, angles, outDir);
%
%  INPUT ARGUMENTS:
%  - imds   : source imageDatastore to augment
%  - angles : vector of rotation angles (degrees) to produce the augmented samples
%  - outDir : directory where the rotated .mat files are saved
%
%  OUTPUT ARGUMENTS:
%  - imdsOut : imageDatastore containing the augmented images, with
%              labels replicated from the source

    if ~exist(outDir, 'dir'), mkdir(outDir); end

    na = numel(angles);
    n  = numel(imds.Files);

    newFiles   = cell(n * na, 1);
    labelsCell = cell(n * na, 1);

    % Generate augmented images
    idx = 0;
    for i = 1:n
        img = readimage(imds, i);
        lbl = imds.Labels(i);

        % Generate an augmented sample for each specified angle
        for a = angles
            idx = idx + 1;
            img_rot = imrotate(img, a, 'crop', 'bilinear');

            fname = fullfile(outDir, sprintf('aug_%04d_%03d.mat', i, a));
            save(fname, 'img_rot');

            newFiles{idx}   = fname;
            labelsCell{idx} = char(lbl);
        end

        % Print progress
        if mod(i, 50) == 0 || i == n
            fprintf('  [%s] %d / %d\n', outDir, i, n);
        end
    end

    % Save labels and resulting datastore
    newLabels = categorical(labelsCell, categories(imds.Labels));
    imdsOut = imageDatastore(newFiles, 'Labels', newLabels, ...
        'FileExtensions', '.mat', ...
        'ReadFcn', @(f) load(f, 'img_rot').img_rot);
end