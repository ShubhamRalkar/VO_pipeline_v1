function [pts_curr, validity, num_tracked] = track_klt_features(...
                                             frame_prev, frame_curr, pts_prev, ~)
% TRACK_KLT_FEATURES  KLT optical flow tracker
% All vision.PointTracker properties hardcoded for Simulink codegen
% klt_params argument kept for API compatibility but ignored

    N = size(pts_prev, 1);

    if N == 0
        pts_curr    = zeros(0, 2);
        validity    = false(0, 1);
        num_tracked = 0;
        return;
    end

    tracker = vision.PointTracker(...
        'BlockSize',             [31 31], ...
        'MaxIterations',         30, ...
        'NumPyramidLevels',      3, ...
        'MaxBidirectionalError', 5.0);

    initialize(tracker, pts_prev, frame_prev);
    [pts_curr, validity] = step(tracker, frame_curr);

    % Enforce image boundary
    W = size(frame_curr, 2);
    H = size(frame_curr, 1);
    oob = pts_curr(:,1) < 1 | pts_curr(:,1) > W | ...
          pts_curr(:,2) < 1 | pts_curr(:,2) > H;
    validity(oob) = false;

    num_tracked = sum(validity);
    release(tracker);
end