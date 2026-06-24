function [road_kp, road_mask, lane_pts, debug_info] = detect_road_features(frame, cam_params, road_params)
% DETECT_ROAD_FEATURES  Detect features specific to road elements
%
% Lightweight pipeline designed for Raspberry Pi 5:
%   1. Adaptive ROI mask (trapezoidal road region)
%   2. Road surface segmentation (intensity-based)
%   3. Lane marking detection (Canny + Hough)
%   4. Road edge / curb detection
%   5. ORB features filtered to road region only
%
% Inputs:
%   frame        (HxW uint8) grayscale image
%   cam_params   struct: .fx .fy .cx .cy .width .height
%   road_params  struct (optional, defaults provided):
%     .roi_top_frac      top of road ROI (fraction of image height, 0.3 = 30%)
%     .roi_bottom_frac   bottom of road ROI (1.0 = full bottom)
%     .roi_top_width     fractional width of ROI at top (0.3 = 30% of image width)
%     .roi_bottom_width  fractional width of ROI at bottom (0.9 = 90%)
%     .road_intensity_low   min intensity for road surface (asphalt)
%     .road_intensity_high  max intensity for road surface
%     .canny_low         Canny low threshold (normalised 0-1)
%     .canny_high        Canny high threshold
%     .hough_threshold   Hough accumulator threshold
%     .hough_min_length  minimum line segment length (px)
%     .hough_max_gap     maximum gap in line segment (px)
%     .lane_angle_min    minimum angle from horizontal for lane lines (deg)
%     .lane_angle_max    maximum angle from horizontal for lane lines (deg)
%     .orb_num_points    max ORB features to detect
%     .morph_radius      morphological cleanup radius (px)
%
% Outputs:
%   road_kp    (Nx2) ORB keypoint locations [u,v] on road only
%   road_mask  (HxW logical) combined road region mask
%   lane_pts   (Mx4) detected lane line segments [x1,y1,x2,y2]
%   debug_info struct with intermediate results for visualisation

    H = size(frame, 1);
    W = size(frame, 2);

    %% ---- Default parameters (tuned for drone altitude ~10m) ----
    if nargin < 3 || isempty(road_params)
        road_params = struct();
    end

    % ROI shape (trapezoid)
    roi_top_frac     = get_field(road_params, 'roi_top_frac',     0.30);
    roi_bottom_frac  = get_field(road_params, 'roi_bottom_frac',  1.00);
    roi_top_width    = get_field(road_params, 'roi_top_width',    0.35);
    roi_bottom_width = get_field(road_params, 'roi_bottom_width', 0.90);

    % Road surface intensity (asphalt is typically mid-gray)
    road_int_low     = get_field(road_params, 'road_intensity_low',  40);
    road_int_high    = get_field(road_params, 'road_intensity_high', 180);

    % Edge / lane detection
    canny_low        = get_field(road_params, 'canny_low',        0.05);
    canny_high       = get_field(road_params, 'canny_high',       0.15);
    hough_threshold  = get_field(road_params, 'hough_threshold',  30);
    hough_min_len    = get_field(road_params, 'hough_min_length', 20);
    hough_max_gap    = get_field(road_params, 'hough_max_gap',    15);
    lane_angle_min   = get_field(road_params, 'lane_angle_min',   20);
    lane_angle_max   = get_field(road_params, 'lane_angle_max',   85);

    % ORB
    orb_num_pts      = get_field(road_params, 'orb_num_points',   300);
    morph_r          = get_field(road_params, 'morph_radius',     5);

    %% ================================================================
    %  STEP 1: Trapezoidal ROI Mask (road region from drone view)
    % ================================================================
    %
    %  In a downward-looking drone view, the road typically forms a
    %  trapezoidal region: narrower at the top (far), wider at bottom (near)
    %
    %       ┌─────────────────────────────┐
    %       │        sky / buildings       │
    %       │     ┌───────────────┐       │  ← roi_top
    %       │    /   ROAD REGION   \      │
    %       │   /                   \     │
    %       │  /                     \    │
    %       │ /                       \   │
    %       │/─────────────────────────\  │  ← roi_bottom
    %       └─────────────────────────────┘

    y_top    = round(roi_top_frac * H);
    y_bottom = round(roi_bottom_frac * H);

    cx = W / 2;    % image center x

    % Top edge of trapezoid
    top_half_w = round(roi_top_width * W / 2);
    x_top_left  = round(cx - top_half_w);
    x_top_right = round(cx + top_half_w);

    % Bottom edge of trapezoid
    bot_half_w = round(roi_bottom_width * W / 2);
    x_bot_left  = round(cx - bot_half_w);
    x_bot_right = round(cx + bot_half_w);

    % Build polygon vertices [x, y]
    roi_poly = [x_top_left,  y_top;
                x_top_right, y_top;
                x_bot_right, y_bottom;
                x_bot_left,  y_bottom];

    roi_mask = poly2mask(roi_poly(:,1), roi_poly(:,2), H, W);

    %% ================================================================
    %  STEP 2: Road Surface Segmentation (intensity-based)
    % ================================================================
    %  Asphalt is mid-gray. Filter by intensity band + smooth.

    % Gaussian blur to reduce noise (lightweight)
    frame_blur = imgaussfilt(frame, 2);

    % Intensity band mask
    intensity_mask = frame_blur >= road_int_low & frame_blur <= road_int_high;

    % Combine with ROI
    road_surface = intensity_mask & roi_mask;

    % Morphological cleanup (close small gaps, remove noise)
    se = strel('disk', morph_r);
    road_surface = imclose(road_surface, se);
    road_surface = imopen(road_surface, strel('disk', max(1, morph_r-2)));

    % Remove small blobs (not road)
    road_surface = bwareaopen(road_surface, 500);

    %% ================================================================
    %  STEP 3: Lane Marking Detection (Canny + Hough)
    % ================================================================
    %  Lane markings are high-contrast edges within the road region.

    % Apply ROI before edge detection (saves computation)
    frame_roi = frame;
    frame_roi(~roi_mask) = 0;

    % Canny edge detection
    edges = edge(frame_roi, 'Canny', [canny_low, canny_high]);

    % Mask edges to road region only
    edges_road = edges & roi_mask;

    % Hough transform for line segments
    [hough_H, theta, rho] = hough(edges_road);
    peaks = houghpeaks(hough_H, 20, 'Threshold', ...
        max(1, ceil(0.3 * max(hough_H(:)))));
    lines = houghlines(edges_road, theta, rho, peaks, ...
        'FillGap', hough_max_gap, 'MinLength', hough_min_len);

    % Filter lines by angle (lane markings are roughly vertical in image)
    lane_pts = zeros(0, 4);
    lane_mask = false(H, W);

    for k = 1:length(lines)
        x1 = lines(k).point1(1); y1 = lines(k).point1(2);
        x2 = lines(k).point2(1); y2 = lines(k).point2(2);

        % Compute angle from horizontal
        dx = x2 - x1;
        dy = y2 - y1;
        angle_deg = abs(atan2d(dy, dx));

        % Keep lines that are roughly along road direction
        if angle_deg >= lane_angle_min && angle_deg <= lane_angle_max
            lane_pts(end+1, :) = [x1, y1, x2, y2]; %#ok<AGROW>

            % Draw thick lane line into mask (widen by 8px for feature region)
            pts_line = bresenham_line(x1, y1, x2, y2);
            for p = 1:size(pts_line, 1)
                px = pts_line(p, 1);
                py = pts_line(p, 2);
                for dr = -4:4
                    for dc = -4:4
                        rr = py + dr;
                        cc = px + dc;
                        if rr >= 1 && rr <= H && cc >= 1 && cc <= W
                            lane_mask(rr, cc) = true;
                        end
                    end
                end
            end
        end
    end

    %% ================================================================
    %  STEP 4: Road Edge / Crosswalk Detection
    % ================================================================
    %  Look for strong horizontal edges near lane regions (crosswalks)
    %  and along road boundaries.

    % Horizontal edge emphasis (crosswalks = horizontal stripes)
    h_kernel = [-1 -1 -1; 0 0 0; 1 1 1];
    h_edges = abs(imfilter(double(frame_roi), h_kernel, 'replicate'));
    crosswalk_mask = h_edges > 30 & roi_mask;
    crosswalk_mask = bwareaopen(crosswalk_mask, 100);

    %% ================================================================
    %  STEP 5: Combined Road Feature Mask
    % ================================================================
    %  Union of: road surface, lane markings, crosswalk areas

    road_mask = road_surface | lane_mask | crosswalk_mask;

    % Ensure within ROI
    road_mask = road_mask & roi_mask;

    % Dilate slightly to catch features near road edges
    road_mask = imdilate(road_mask, strel('disk', 3));

    %% ================================================================
    %  STEP 6: Detect ORB Features Within Road Mask Only
    % ================================================================
    %  Apply mask to frame, then detect ORB.

    % Create masked frame (zero out non-road pixels)
    frame_masked = frame;
    frame_masked(~road_mask) = 0;

    % Detect ORB on masked frame
    orb_points = detectORBFeatures(frame_masked, ...
        'NumLevels',   8, ...
        'ScaleFactor', 1.2);

    orb_points = orb_points.selectStrongest(orb_num_pts);
    valid_pts = orb_points.Location;   % Nx2

    % Final filter: keep only points strictly inside road mask
    in_mask = false(size(valid_pts, 1), 1);
    for i = 1:size(valid_pts, 1)
        px = round(valid_pts(i, 1));
        py = round(valid_pts(i, 2));
        if px >= 1 && px <= W && py >= 1 && py <= H
            in_mask(i) = road_mask(py, px);
        end
    end
    road_kp = valid_pts(in_mask, :);

    %% ---- Debug info for visualisation ----
    debug_info.roi_mask       = roi_mask;
    debug_info.roi_poly       = roi_poly;
    debug_info.intensity_mask = intensity_mask;
    debug_info.road_surface   = road_surface;
    debug_info.edges_road     = edges_road;
    debug_info.lane_mask      = lane_mask;
    debug_info.crosswalk_mask = crosswalk_mask;
    debug_info.n_lanes        = size(lane_pts, 1);
    debug_info.n_features     = size(road_kp, 1);
    debug_info.frame_masked   = frame_masked;
end

%% ================ Helper Functions ================

function val = get_field(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end

function pts = bresenham_line(x1, y1, x2, y2)
    % Simple Bresenham line rasteriser — returns Nx2 [x,y] pixel coords
    dx = abs(x2 - x1);
    dy = abs(y2 - y1);
    n = max(dx, dy) + 1;
    pts = zeros(n, 2);

    x = x1; y = y1;
    sx = sign(x2 - x1);
    sy = sign(y2 - y1);

    if dx >= dy
        err = dx / 2;
        for i = 1:n
            pts(i,:) = [round(x), round(y)];
            x = x + sx;
            err = err - dy;
            if err < 0
                y = y + sy;
                err = err + dx;
            end
        end
    else
        err = dy / 2;
        for i = 1:n
            pts(i,:) = [round(x), round(y)];
            y = y + sy;
            err = err - dx;
            if err < 0
                x = x + sx;
                err = err + dy;
            end
        end
    end
end
