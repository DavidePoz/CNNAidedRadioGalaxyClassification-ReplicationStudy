function mask = lowNoiseThresh(img, s)
%LOWNOISETHRESH - Thresholding for images with low pixel noise.
%
%  SYNTAX:
%  mask = lowNoiseThresh(img, s);
%
%  INPUT ARGUMENTS:
%  - img : fits image, normalized in [0, 1]
%  - s   : number of histogram bins with more than 100 pixels
%
%  OUTPUT ARGUMENTS:
%  - mask : binary mask highlighting the galaxy pixels

    if s <= 14
        % Very low noise: static thresholding
        mask = img > 0.1;
    else
        % Quantile-based thresholding
        q985 = quantile(img(:), 0.985);
        q98  = quantile(img(:), 0.98);
        
        % Extract brightest pixels and remove small clusters
        first = img > q985;
        first = bwareaopen(first, 10);
        
        % Extended mask
        exten = img >= q98;
        se = strel('square', 3);
        
        % Grow first proposal to include galaxy's edge pixels
        prev = false(size(first)); 
        cur = first;
        
        while any(cur(:) ~= prev(:))
            prev = cur;
            cur = imdilate(cur, se) & exten;
        end
        mask = cur;
    end
    
    % Remove leftover small clusters
    mask = bwareaopen(mask, 5);
end