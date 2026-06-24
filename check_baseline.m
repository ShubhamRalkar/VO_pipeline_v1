function [ready, distance] = check_baseline(pos_0, pos_curr, min_baseline)
% CHECK_BASELINE  Check if drone has moved far enough to initialise map
%
% Inputs:
%   pos_0        (3×1) position at first keyframe
%   pos_curr     (3×1) current drone position
%   min_baseline scalar minimum required displacement [m]
%
% Outputs:
%   ready    logical — true if baseline >= min_baseline
%   distance scalar  — current baseline distance [m]

distance = norm(pos_curr - pos_0);
ready    = (distance >= min_baseline);
end