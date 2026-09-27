
function [img_std, angle_deg] = stdRotation(thresh_img, img)
%STDROTATION - Standardizes image rotation with SVD.
%
%  SYNTAX:
%  [img_std, angle] = stdRotation(thresh_img, img);
%
%  INPUT ARGUMENTS:
%  - thresh_img : thresholded version of the image to rotate
%  - img : image to rotate
%
%  OUTPUT ARGUMENTS:
%  - img_std : standardized rotation of the input image
%  - angle_deg : rotation angle applied, in degrees 

    % Images whose masks contain a small number of pixels are left as they are
    if sum(thresh_img(:)) < 20
        img_std   = img;
        angle_deg = 0;
        return;
    end

    % Extract coordinates of thresholded pixels
    [rows, cols] = find(thresh_img);
    X = [cols.'; rows.'];                   % X shape: 2 x P

    % SV decomposition on centred pixels
    X_centered = X - mean(X, 2);
    [U, ~, ~] = svd(X_centered, 'econ');    % U shape: 2 x 2
    
    % Compute potential rotation angle
    candidate_theta = asin(U(2,1));

    % Check if U should be reflected
    cos_theta = cos(candidate_theta);
    
    if sign(cos_theta) ~= sign(U(1,1))
        U(:,1) = -U(:,1);
    end
    if sign(cos_theta) ~= sign(U(2,2))
        U(:,2) = -U(:,2);
    end
    
    U_T = transpose(U);
    
    % Determine rotation direction
    U_T(1,2) = U_T(1,2) / sin(candidate_theta);
    U_T(2,1) = U_T(2,1) / sin(candidate_theta);

    if U_T(1,2) < 0
        theta = candidate_theta;
    else
        theta = - candidate_theta;
    end
    
    % Rotate original image
    angle_deg = rad2deg(theta);
    img_std = imrotate(img, -angle_deg, 'crop', 'bilinear');
end