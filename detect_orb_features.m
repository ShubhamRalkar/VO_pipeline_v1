function [keypoints, descriptors, num_detected] = detect_orb_features(frame, orb_params)
% DETECT_ORB_FEATURES  Detect ORB keypoints and extract descriptors
%
% Inputs:
%   frame      (H×W uint8) grayscale image
%   orb_params struct with fields:
%              .num_points   max features to keep (strongest first)
%              .num_levels   pyramid levels
%              .scale_factor scale between pyramid levels
%
% Outputs:
%   keypoints     (Nx2 double) [u, v] pixel coordinates
%   descriptors   (Nx32 uint8) binary descriptors (256 bits each)
%   num_detected  (scalar) number of features found

    % Detect ORB keypoints — older toolbox versions don't support NumPoints
    orb_points = detectORBFeatures(frame, ...
        'NumLevels',   orb_params.num_levels, ...
        'ScaleFactor', orb_params.scale_factor);

    % Keep only the strongest N keypoints
    orb_points = orb_points.selectStrongest(orb_params.num_points);

    % Extract descriptors at detected locations
    [features, valid_points] = extractFeatures(frame, orb_points, ...
        'Method', 'ORB');

    % Extract [u, v] coordinates
    keypoints    = valid_points.Location;   % Nx2 double
    descriptors  = features.Features;       % Nx32 uint8
    num_detected = size(keypoints, 1);
end