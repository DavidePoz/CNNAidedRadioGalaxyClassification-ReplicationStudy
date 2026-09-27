function mask = thresholdImage(img)
%THRESHOLDIMAGE - Applies thresholding to the input image to extract its galaxy pixels.
%
%  SYNTAX:
%  mask = thresholdImage(img);
%
%  INPUT ARGUMENTS:
%  - img : fits image, normalized in [0, 1]
%
%  OUTPUT ARGUMENTS:
%  - mask : binary mask highlighting the galaxy pixels

    % Compute metrics to determine noise level
    std_val = std(img(:));
    [counts, ~] = histcounts(img, 256);
    s = sum(counts > 100);
    
    % Handle low and high noise levels
    if s < 17 || std_val < 0.035
        mask = lowNoiseThresh(img, s);
    else
        mask = highNoiseThresh(img);
    end
end