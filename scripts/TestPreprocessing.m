%% TEST PRE-PROCESSING FUNCTIONS
%  This script only serves as a test to verify the 
%  correct working of the pre-processing functions

clc; close all;

% Load datastores
load(fullfile('data', 'datastores.mat'))

% Sample 3 random images
idx_samples = randperm(length(imdsTrain.Files), 3);

% Plot sampled images and results of pre-processing steps
figure('Name', 'Test Preprocessing', 'NumberTitle', 'off', 'Position', [100 100 1200 400]);
for i = 1:3

    img = readimage(imdsTrain, idx_samples(i));
    label = imdsTrain.Labels(idx_samples(i));

    thresh_img = thresholdImage(img);
    [img_std, angle] = stdRotation(thresh_img, img);
    
    subplot(3, 3, i); 
    imshow(img, []); 
    title(sprintf('Original (class: %s)', char(label)));
    
    subplot(3,3,i+3);
    imshow(thresh_img, []);
    title(sprintf('Thresholded image'));

    subplot(3, 3, i+6); 
    imshow(img_std, []); 
    title(sprintf('Standardized image (%.1f° rotation)', -angle));
end
sgtitle('Original - Thresholded - Standardized images');