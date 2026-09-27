function mask = highNoiseThresh(img)
%HIGHNOISETHRESH - Thresholding for images with significant pixel noise.
%
%  SYNTAX:
%  mask = highNoiseThresh(img);
%
%  INPUT ARGUMENTS:
%  - img : fits image, normalized in [0, 1]
%
%  OUTPUT ARGUMENTS:
%  - mask : binary mask highlighting the galaxy pixels

    % Compute the distribution of unique values in 9x9 neighbourhoods
    val_dist = findNumVals(img);
    
    % Extract quantiles
    q985 = quantile(val_dist(:), 0.985);
    q96  = quantile(val_dist(:), 0.96);
    
    % Pixels along the galaxy border have a wider variety of values
    first = val_dist >= q985;
    % Remove small artefacts before dilation (thresh=10)
    first = bwareaopen(first, 10);
    
    % Extended mask (limit for the growth)
    exten = val_dist >= q96;
    se = strel('square', 3);
    
    % Initialize prev as all-false to force the loop to run at least once
    prev = false(size(first));
    cur  = first;
    
    while any(cur(:) ~= prev(:))
        prev = cur;
        cur  = imdilate(cur, se) & exten;
    end
    
    % Final cleanup (thresh=5) and fill holes
    mask = bwareaopen(cur, 5);
    mask = imfill(mask, 'holes');
end