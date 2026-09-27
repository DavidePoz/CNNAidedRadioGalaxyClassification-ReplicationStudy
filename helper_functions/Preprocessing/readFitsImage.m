function img = readFitsImage(filename)
%READFITSIMAGE - Reads a fits image, centre-crops it to size 150x150, and normalises its pixels to [0,1].
%
%  SYNTAX:
%  img = readFitsImage(filename)
%
%  INPUT ARGUMENTS:
%  - filename : image file to be read
%
%  OUTPUT ARGUMENTS:
%  - img : normalised version of the input image
    
    % Read image and focus it
    rawImg = fitsread(filename, 'primary');
    if iscell(rawImg), rawImg = rawImg{1}; end
    rawImg = double(rawImg);
    focus = rawImg(76:225, 76:225);

    % Normalize image values
    minVal = min(focus(:));
    maxVal = max(focus(:));
    if maxVal > minVal
        img = (focus - minVal) / (maxVal - minVal);
    else
        img = zeros(size(focus));
    end

end