function [pts_prev_in, pts_curr_in, F, num_inliers] = ransac_filter(...
    pts_prev, pts_curr, ransac_params)
% RANSAC_FILTER  Remove outlier tracks using epipolar geometry (RANSAC)
%
% Inputs:
%   pts_prev      (Nx2 double) keypoint positions in frame_prev
%   pts_curr      (Nx2 double) tracked positions  in frame_curr
%   ransac_params struct with fields:
%                 .max_distance   epipolar line distance threshold (pixels)
%                 .confidence     RANSAC confidence (%)
%                 .max_iterations max RANSAC iterations
%
% Outputs:
%   pts_prev_in   (Mx2) inlier points in frame_prev  (M <= N)
%   pts_curr_in   (Mx2) inlier points in frame_curr
%   F             (3×3) Fundamental matrix
%   num_inliers   scalar

% Need at least 8 point pairs for the 8-point algorithm
MIN_POINTS = 8;

if size(pts_prev, 1) < MIN_POINTS
    pts_prev_in = pts_prev;
    pts_curr_in = pts_curr;
    F           = eye(3);
    num_inliers = size(pts_prev, 1);
    warning('ransac_filter: fewer than 8 points — skipping RANSAC');
    return;
end

% Estimate Fundamental Matrix with RANSAC
[F, inlier_mask] = estimateFundamentalMatrix(pts_prev, pts_curr, ...
    'Method',        'RANSAC', ...
    'NumTrials',     ransac_params.max_iterations, ...
    'DistanceThreshold', ransac_params.max_distance, ...
    'Confidence',    ransac_params.confidence);

% Apply inlier mask
pts_prev_in = pts_prev(inlier_mask, :);
pts_curr_in = pts_curr(inlier_mask, :);
num_inliers = size(pts_prev_in, 1);
end