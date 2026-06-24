%% demo_road_features.m — Demonstration of Road-Specific Feature Detection
% =========================================================================
%  Tests the detect_road_features.m pipeline on real-world drone imagery
%  to show how it isolates road surfaces, lanes, and crosswalks while
%  ignoring trees, sky, and buildings.
%
%  Run from: C:\DroneSimulation\Map_initialiser
% =========================================================================

clear; clc; clear functions;
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║    ROAD-SPECIFIC FEATURE DETECTION DEMO                      ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% ---- Load parameters ----
run('drone_params.m');

cam_p.fx     = cam_fx;
cam_p.fy     = cam_fy;
cam_p.cx     = cam_cx;
cam_p.cy     = cam_cy;
cam_p.width  = frame_width;
cam_p.height = frame_height;

% Custom road parameters
road_p.roi_top_frac     = 0.35;  % Start looking 35% down from top
road_p.roi_bottom_frac  = 1.0;
road_p.roi_top_width    = 0.4;
road_p.roi_bottom_width = 1.0;
road_p.orb_num_points   = 500;

%% ---- Load Images ----
fprintf('Loading images...\n');
img_dir = fileparts(mfilename('fullpath'));
if isempty(img_dir), img_dir = pwd; end

files = {'road_frame0.png', 'road_frame1.png', 'road_frame2.png'};
frames = cell(1,3);
rgb_imgs = cell(1,3);

for i = 1:3
    img_path = fullfile(img_dir, files{i});
    rgb = imread(img_path);
    if size(rgb, 3) == 3
        gray = rgb2gray(rgb);
    else
        gray = rgb;
    end
    
    gray = imresize(gray, [frame_height, frame_width]);
    rgb = imresize(rgb, [frame_height, frame_width]);
    
    frames{i} = gray;
    rgb_imgs{i} = rgb;
end

%% ---- Process Images ----
fig = figure('Name', 'Road-Specific Features', 'Position', [50 100 1500 450], 'Color', [0.06 0.06 0.1]);

for i = 1:3
    fprintf('\nProcessing Frame %d...\n', i-1);
    tic;
    [road_kp, road_mask, lane_pts, debug] = detect_road_features(frames{i}, cam_p, road_p);
    t_proc = toc;
    
    fprintf('  Found %d road features in %.1f ms\n', size(road_kp, 1), t_proc*1000);
    fprintf('  Detected %d lane segments\n', size(lane_pts, 1));
    
    % Visualisation
    subplot(1, 3, i);
    
    % Create overlay: Tint non-road areas dark blue, keep road bright
    rgb = rgb_imgs{i};
    overlay = rgb;
    blue_tint = overlay(:,:,3);
    blue_tint(~road_mask) = min(255, blue_tint(~road_mask) + 50);
    overlay(:,:,3) = blue_tint;
    
    red_tint = overlay(:,:,1);
    red_tint(~road_mask) = max(0, red_tint(~road_mask) - 50);
    overlay(:,:,1) = red_tint;
    
    green_tint = overlay(:,:,2);
    green_tint(~road_mask) = max(0, green_tint(~road_mask) - 50);
    overlay(:,:,2) = green_tint;
    
    imshow(overlay); hold on;
    
    % Plot ROI boundary
    plot(debug.roi_poly([1:end 1],1), debug.roi_poly([1:end 1],2), 'y--', 'LineWidth', 1);
    
    % Plot lane lines
    for k = 1:size(lane_pts, 1)
        plot([lane_pts(k,1), lane_pts(k,3)], [lane_pts(k,2), lane_pts(k,4)], 'r-', 'LineWidth', 2);
    end
    
    % Plot keypoints
    if ~isempty(road_kp)
        scatter(road_kp(:,1), road_kp(:,2), 30, 'g', 'filled', 'MarkerEdgeColor', 'w');
    end
    
    title(sprintf('Frame %d: %d Road Features', i-1, size(road_kp,1)), 'Color', 'w', 'FontSize', 12);
    if i == 1
        legend({'ROI Boundary', 'Detected Lanes', 'Road ORB Features'}, 'TextColor', 'w', 'Color', 'k', 'Location', 'southwest');
    end
    hold off;
end

sgtitle('Road-Specific Feature Detection (Trapezoidal ROI + Segmentation + Lane Detection)', 'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold');
fprintf('\n✓ Demo complete. Figure generated.\n');
