%% demo_vo_real_world.m — VO Pipeline Demo on Real-World Road Images
% =========================================================================
%  Tests the VO front-end modules on real-world drone imagery:
%
%   Test 1: ORB Feature Detection on 3 real road scenes
%   Test 2: KLT Optical Flow Tracking (Frame 0 → Frame 1, Frame 0 → Frame 2)
%   Test 3: RANSAC Epipolar Filtering
%   Test 4: Essential Matrix & Relative Pose Recovery
%   Test 5: Triangulation (Structure from Motion) between frame pairs
%
%  Images: Drone aerial views of a suburban road with cars, trees, buildings
%  Run from:  C:\DroneSimulation\Map_initialiser
% =========================================================================

clear; clc; clear functions;
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║    VISUAL ODOMETRY — REAL-WORLD IMAGE DEMONSTRATION          ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% ---- Load parameters for camera model ----
run('drone_params.m');

orb_p.num_points   = orb_num_points;
orb_p.num_levels   = orb_num_levels;
orb_p.scale_factor = orb_scale_factor;

klt_p.block_size     = klt_block_size;
klt_p.max_iterations = klt_max_iterations;
klt_p.num_levels     = klt_num_levels;
klt_p.max_error      = klt_max_error;

ran_p.max_distance   = ransac_max_distance;
ran_p.confidence     = ransac_confidence;
ran_p.max_iterations = ransac_max_iterations;

results = struct('test', {}, 'pass', {}, 'detail', {});

% Pre-initialise variables that may be skipped by conditional blocks
flow_mag_01 = 0;
flow_mag_02 = 0;
n_valid_tri = 0;
best_t = zeros(3,1);
best_R = eye(3);

%% ---- Load real-world images ----
fprintf('Loading real-world road images...\n');

img_dir = fileparts(mfilename('fullpath'));
if isempty(img_dir), img_dir = pwd; end

img0_path = fullfile(img_dir, 'road_frame0.png');
img1_path = fullfile(img_dir, 'road_frame1.png');
img2_path = fullfile(img_dir, 'road_frame2.png');

assert(isfile(img0_path), 'road_frame0.png not found in %s', img_dir);
assert(isfile(img1_path), 'road_frame1.png not found in %s', img_dir);
assert(isfile(img2_path), 'road_frame2.png not found in %s', img_dir);

img0_rgb = imread(img0_path);
img1_rgb = imread(img1_path);
img2_rgb = imread(img2_path);

% Convert to grayscale (VO pipeline works on grayscale)
frame_0 = rgb2gray_safe(img0_rgb);
frame_1 = rgb2gray_safe(img1_rgb);
frame_2 = rgb2gray_safe(img2_rgb);

% Resize to match camera model (640x480) if needed
target_w = frame_width;
target_h = frame_height;
frame_0 = imresize(frame_0, [target_h, target_w]);
frame_1 = imresize(frame_1, [target_h, target_w]);
frame_2 = imresize(frame_2, [target_h, target_w]);

% Keep RGB versions for display (also resized)
img0_disp = imresize(img0_rgb, [target_h, target_w]);
img1_disp = imresize(img1_rgb, [target_h, target_w]);
img2_disp = imresize(img2_rgb, [target_h, target_w]);

fprintf('  Images loaded and resized to %dx%d\n', target_w, target_h);
fprintf('  Frame 0: min=%d max=%d mean=%.0f\n', min(frame_0(:)), max(frame_0(:)), mean(double(frame_0(:))));
fprintf('  Frame 1: min=%d max=%d mean=%.0f\n', min(frame_1(:)), max(frame_1(:)), mean(double(frame_1(:))));
fprintf('  Frame 2: min=%d max=%d mean=%.0f\n', min(frame_2(:)), max(frame_2(:)), mean(double(frame_2(:))));

%% ================================================================
%  TEST 1: ORB Feature Detection on Real Images
% ================================================================
fprintf('\n┌──────────────────────────────────────────────────┐\n');
fprintf('│  TEST 1: ORB Feature Detection on Road Scenes   │\n');
fprintf('└──────────────────────────────────────────────────┘\n');

tic;
[kp_0, desc_0, n0] = detect_orb_features(frame_0, orb_p);
t0 = toc;

tic;
[kp_1, desc_1, n1] = detect_orb_features(frame_1, orb_p);
t1 = toc;

tic;
[kp_2, desc_2, n2] = detect_orb_features(frame_2, orb_p);
t2 = toc;

fprintf('  Frame 0: %d ORB features  (%.1f ms)\n', n0, t0*1000);
fprintf('  Frame 1: %d ORB features  (%.1f ms)\n', n1, t1*1000);
fprintf('  Frame 2: %d ORB features  (%.1f ms)\n', n2, t2*1000);

orb_ok = n0 >= 30 && n1 >= 30 && n2 >= 30;
fprintf('  All frames ≥30 features: %s\n', pass_fail(orb_ok));

% Spatial distribution analysis
quadrants_0 = analyze_spatial(kp_0, target_w, target_h);
fprintf('  Frame 0 feature spread: TL=%d TR=%d BL=%d BR=%d\n', ...
    quadrants_0(1), quadrants_0(2), quadrants_0(3), quadrants_0(4));
spread_ok = all(quadrants_0 > 0);
fprintf('  Features in all quadrants: %s\n', pass_fail(spread_ok));

results(end+1) = struct('test', 'ORB detects ≥30 on real images', 'pass', orb_ok, ...
    'detail', sprintf('F0=%d F1=%d F2=%d', n0, n1, n2));
results(end+1) = struct('test', 'Features spatially distributed', 'pass', spread_ok, ...
    'detail', sprintf('[%d,%d,%d,%d]', quadrants_0));

%% ================================================================
%  TEST 2: KLT Tracking Between Real Frames
% ================================================================
fprintf('\n┌──────────────────────────────────────────────────┐\n');
fprintf('│  TEST 2: KLT Optical Flow on Road Scenes        │\n');
fprintf('└──────────────────────────────────────────────────┘\n');

% Track Frame 0 → Frame 1 (small motion)
tic;
[kp_01, val_01, n_t01] = track_klt_features(frame_0, frame_1, kp_0, klt_p);
t_klt01 = toc;
ratio_01 = n_t01 / n0 * 100;

% Track Frame 0 → Frame 2 (larger motion)
tic;
[kp_02, val_02, n_t02] = track_klt_features(frame_0, frame_2, kp_0, klt_p);
t_klt02 = toc;
ratio_02 = n_t02 / n0 * 100;

fprintf('  Frame 0 → 1: %d/%d tracked (%.0f%%)  %.1f ms\n', n_t01, n0, ratio_01, t_klt01*1000);
fprintf('  Frame 0 → 2: %d/%d tracked (%.0f%%)  %.1f ms\n', n_t02, n0, ratio_02, t_klt02*1000);

% Compute flow statistics
if n_t01 > 0
    flow_01 = kp_01(val_01,:) - kp_0(val_01,:);
    flow_mag_01 = sqrt(sum(flow_01.^2, 2));
    fprintf('  Flow 0→1: mean=%.1f px, max=%.1f px, std=%.1f px\n', ...
        mean(flow_mag_01), max(flow_mag_01), std(flow_mag_01));
end

if n_t02 > 0
    flow_02 = kp_02(val_02,:) - kp_0(val_02,:);
    flow_mag_02 = sqrt(sum(flow_02.^2, 2));
    fprintf('  Flow 0→2: mean=%.1f px, max=%.1f px, std=%.1f px\n', ...
        mean(flow_mag_02), max(flow_mag_02), std(flow_mag_02));
end

klt_ok_01 = n_t01 >= 15;
klt_ok_02 = n_t02 >= 10;
fprintf('  0→1 tracks ≥15: %s\n', pass_fail(klt_ok_01));
fprintf('  0→2 tracks ≥10: %s\n', pass_fail(klt_ok_02));

results(end+1) = struct('test', 'KLT 0→1 tracks ≥15', 'pass', klt_ok_01, ...
    'detail', sprintf('%d tracked (%.0f%%)', n_t01, ratio_01));
results(end+1) = struct('test', 'KLT 0→2 tracks ≥10', 'pass', klt_ok_02, ...
    'detail', sprintf('%d tracked (%.0f%%)', n_t02, ratio_02));

%% ================================================================
%  TEST 3: RANSAC Epipolar Filtering
% ================================================================
fprintf('\n┌──────────────────────────────────────────────────┐\n');
fprintf('│  TEST 3: RANSAC Outlier Filtering               │\n');
fprintf('└──────────────────────────────────────────────────┘\n');

if n_t01 >= 8
    tic;
    [p0_in_01, p1_in_01, F_01, n_in_01] = ransac_filter(...
        kp_0(val_01,:), kp_01(val_01,:), ran_p);
    t_r01 = toc;
    inlier_ratio_01 = n_in_01 / n_t01 * 100;
    fprintf('  0→1 RANSAC: %d/%d inliers (%.0f%%)  %.1f ms\n', ...
        n_in_01, n_t01, inlier_ratio_01, t_r01*1000);
    fprintf('  F matrix rank: %d (expected 2)\n', rank(F_01));
    r01_ok = n_in_01 >= 8;
else
    fprintf('  0→1 RANSAC: SKIPPED (too few tracks)\n');
    r01_ok = false;
    n_in_01 = 0; p0_in_01 = []; p1_in_01 = []; F_01 = eye(3);
end

if n_t02 >= 8
    tic;
    [p0_in_02, p2_in_02, F_02, n_in_02] = ransac_filter(...
        kp_0(val_02,:), kp_02(val_02,:), ran_p);
    t_r02 = toc;
    inlier_ratio_02 = n_in_02 / n_t02 * 100;
    fprintf('  0→2 RANSAC: %d/%d inliers (%.0f%%)  %.1f ms\n', ...
        n_in_02, n_t02, inlier_ratio_02, t_r02*1000);
    fprintf('  F matrix rank: %d (expected 2)\n', rank(F_02));
    r02_ok = n_in_02 >= 8;
else
    fprintf('  0→2 RANSAC: SKIPPED (too few tracks)\n');
    r02_ok = false;
    n_in_02 = 0; p0_in_02 = []; p2_in_02 = []; F_02 = eye(3);
end

fprintf('  0→1 inliers ≥8: %s\n', pass_fail(r01_ok));
fprintf('  0→2 inliers ≥8: %s\n', pass_fail(r02_ok));

results(end+1) = struct('test', 'RANSAC 0→1 inliers ≥8', 'pass', r01_ok, ...
    'detail', sprintf('%d inliers', n_in_01));
results(end+1) = struct('test', 'RANSAC 0→2 inliers ≥8', 'pass', r02_ok, ...
    'detail', sprintf('%d inliers', n_in_02));

%% ================================================================
%  TEST 4: Essential Matrix & Relative Pose Recovery
% ================================================================
fprintf('\n┌──────────────────────────────────────────────────┐\n');
fprintf('│  TEST 4: Essential Matrix & Pose Recovery        │\n');
fprintf('└──────────────────────────────────────────────────┘\n');

intrinsics = cameraIntrinsics([cam_fx, cam_fy], [cam_cx, cam_cy], [target_h, target_w]);

ess_ok = false;
pose_ok = false;

if r01_ok && n_in_01 >= 8
    try
        [E_01, ess_inliers_01] = estimateEssentialMatrix(...
            p0_in_01, p1_in_01, intrinsics, ...
            'Confidence', 99.0, 'MaxNumTrials', 1000, 'MaxDistance', 1.5);
        n_ess_in = sum(ess_inliers_01);
        ess_ok = n_ess_in >= 5;
        fprintf('  Essential Matrix (0→1): %d inliers %s\n', n_ess_in, pass_fail(ess_ok));

        if ess_ok
            % Decompose E into R, t
            [U_e, ~, V_e] = svd(E_01);
            E_clean = U_e * diag([1,1,0]) * V_e';
            [U_e, ~, V_e] = svd(E_clean);

            W_mat = [0 -1 0; 1 0 0; 0 0 1];
            R1 = U_e * W_mat * V_e';
            R2 = U_e * W_mat' * V_e';
            t_pos = U_e(:,3);

            if det(R1) < 0, R1 = -R1; end
            if det(R2) < 0, R2 = -R2; end

            % Test all 4 candidates via cheirality
            K = [cam_fx 0 cam_cx; 0 cam_fy cam_cy; 0 0 1];
            P0 = K * [eye(3), zeros(3,1)];

            best_count = 0;
            best_R = eye(3);
            best_t = zeros(3,1);

            cands = {R1, t_pos; R1, -t_pos; R2, t_pos; R2, -t_pos};
            for c = 1:4
                R_c = cands{c,1};
                t_c = cands{c,2};
                PK_c = K * [R_c, t_c];
                count = 0;
                pts_0_ess = p0_in_01(ess_inliers_01,:);
                pts_1_ess = p1_in_01(ess_inliers_01,:);
                for i = 1:min(size(pts_0_ess,1), 50)
                    A = [pts_0_ess(i,1)*P0(3,:)-P0(1,:);
                         pts_0_ess(i,2)*P0(3,:)-P0(2,:);
                         pts_1_ess(i,1)*PK_c(3,:)-PK_c(1,:);
                         pts_1_ess(i,2)*PK_c(3,:)-PK_c(2,:)];
                    [~,~,Vt] = svd(A);
                    X_h = Vt(:,end);
                    if abs(X_h(4)) < 1e-10, continue; end
                    X = X_h(1:3)/X_h(4);
                    XK = R_c*X + t_c;
                    if X(3) > 0 && XK(3) > 0
                        count = count + 1;
                    end
                end
                if count > best_count
                    best_count = count;
                    best_R = R_c;
                    best_t = t_c;
                end
            end

            % Convert R to euler for display
            pitch_r = -asin(max(-1, min(1, best_R(3,1))));
            cp_r = cos(pitch_r);
            if abs(cp_r) > 1e-6
                roll_r = atan2(best_R(3,2)/cp_r, best_R(3,3)/cp_r);
                yaw_r  = atan2(best_R(2,1)/cp_r, best_R(1,1)/cp_r);
            else
                roll_r = 0; yaw_r = atan2(-best_R(1,2), best_R(2,2));
            end

            pose_ok = best_count >= 3;
            fprintf('  Relative pose (best candidate): %d/%d points in front %s\n', ...
                best_count, min(n_ess_in, 50), pass_fail(pose_ok));
            fprintf('  Translation dir: [%.3f, %.3f, %.3f]\n', best_t(1), best_t(2), best_t(3));
            fprintf('  Rotation (deg):  roll=%.1f° pitch=%.1f° yaw=%.1f°\n', ...
                rad2deg(roll_r), rad2deg(pitch_r), rad2deg(yaw_r));
        end
    catch ME
        fprintf('  Essential Matrix estimation failed: %s\n', ME.message);
    end
else
    fprintf('  SKIPPED: not enough RANSAC inliers for Essential Matrix\n');
end

results(end+1) = struct('test', 'Essential Matrix computed', 'pass', ess_ok, ...
    'detail', '');
results(end+1) = struct('test', 'Relative pose recovered', 'pass', pose_ok, ...
    'detail', '');

%% ================================================================
%  TEST 5: Triangulation (3D Reconstruction)
% ================================================================
fprintf('\n┌──────────────────────────────────────────────────┐\n');
fprintf('│  TEST 5: Triangulation (3D Structure)            │\n');
fprintf('└──────────────────────────────────────────────────┘\n');

tri_ok = false;
tri_points = [];
n_valid_tri = 0;

if ess_ok && pose_ok
    P0_tri = K * [eye(3), zeros(3,1)];
    PK_tri = K * [best_R, best_t];

    pts_0_tri = p0_in_01(ess_inliers_01,:);
    pts_1_tri = p1_in_01(ess_inliers_01,:);
    N_tri = size(pts_0_tri, 1);

    tri_points = zeros(N_tri, 3);
    n_valid_tri = 0;

    for i = 1:N_tri
        A = [pts_0_tri(i,1)*P0_tri(3,:) - P0_tri(1,:);
             pts_0_tri(i,2)*P0_tri(3,:) - P0_tri(2,:);
             pts_1_tri(i,1)*PK_tri(3,:) - PK_tri(1,:);
             pts_1_tri(i,2)*PK_tri(3,:) - PK_tri(2,:)];
        [~,~,V_svd] = svd(A);
        X_h = V_svd(:,end);
        if abs(X_h(4)) < 1e-10, continue; end
        X = X_h(1:3) / X_h(4);

        % Depth in both cameras must be positive
        X_cam2 = best_R * X + best_t;
        if X(3) > 0 && X_cam2(3) > 0
            n_valid_tri = n_valid_tri + 1;
            tri_points(n_valid_tri, :) = X';
        end
    end
    tri_points = tri_points(1:n_valid_tri, :);

    tri_ok = n_valid_tri >= 5;
    fprintf('  Triangulated: %d / %d valid 3D points %s\n', n_valid_tri, N_tri, pass_fail(tri_ok));

    if n_valid_tri > 0
        fprintf('  Depth range: %.2f to %.2f (camera units)\n', ...
            min(tri_points(:,3)), max(tri_points(:,3)));
        fprintf('  X range:     %.2f to %.2f\n', min(tri_points(:,1)), max(tri_points(:,1)));
        fprintf('  Y range:     %.2f to %.2f\n', min(tri_points(:,2)), max(tri_points(:,2)));
    end
else
    fprintf('  SKIPPED: Essential matrix or pose not available\n');
end

results(end+1) = struct('test', 'Triangulation ≥5 3D points', 'pass', tri_ok, ...
    'detail', sprintf('%d points', n_valid_tri));

%% ================================================================
%  RESULTS SUMMARY
% ================================================================
fprintf('\n╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                  REAL-WORLD TEST RESULTS                      ║\n');
fprintf('╠════════════════════════════════════════════════════════════════╣\n');

n_pass = 0; n_fail = 0;
for i = 1:length(results)
    if results(i).pass
        status = '✓ PASS';
        n_pass = n_pass + 1;
    else
        status = '✗ FAIL';
        n_fail = n_fail + 1;
    end
    d = results(i).detail;
    if ~isempty(d), d = [' (' d ')']; end
    fprintf('║  %-42s %s%s\n', results(i).test, status, d);
end

fprintf('╠════════════════════════════════════════════════════════════════╣\n');
fprintf('║  TOTAL: %d passed, %d failed out of %d tests                 ║\n', ...
    n_pass, n_fail, n_pass+n_fail);
if n_fail == 0
    fprintf('║  STATUS: ALL TESTS PASSED ✓                                  ║\n');
else
    fprintf('║  STATUS: %d TEST(S) FAILED ✗                                 ║\n', n_fail);
end
fprintf('╚════════════════════════════════════════════════════════════════╝\n');

%% ================================================================
%  VISUALISATION — 4 Rich Figures
% ================================================================
fprintf('\nGenerating visualisation figures...\n');

% ===== FIGURE 1: ORB Features on All 3 Frames =====
fig1 = figure('Name', 'Real-World VO: ORB Features', ...
    'Position', [50 550 1400 420], 'Color', [0.06 0.06 0.10]);

for idx = 1:3
    subplot(1,3,idx);
    switch idx
        case 1; img = img0_disp; kp = kp_0; nd = n0; lbl = 'Frame 0';
        case 2; img = img1_disp; kp = kp_1; nd = n1; lbl = 'Frame 1';
        case 3; img = img2_disp; kp = kp_2; nd = n2; lbl = 'Frame 2';
    end
    imshow(img); hold on;
    scatter(kp(:,1), kp(:,2), 30, 'g', 'LineWidth', 1.2);
    title(sprintf('%s: %d ORB features', lbl, nd), 'Color', 'w', 'FontSize', 12);
    hold off;
end
sgtitle('Test 1: ORB Feature Detection on Road Scenes', 'Color', 'w', 'FontSize', 15, 'FontWeight', 'bold');

% ===== FIGURE 2: KLT Tracking + Optical Flow =====
fig2 = figure('Name', 'Real-World VO: KLT Tracking', ...
    'Position', [50 100 1400 500], 'Color', [0.06 0.06 0.10]);

% Panel 1: Flow 0→1
subplot(1,2,1);
imshow(img1_disp); hold on;
if n_t01 > 0
    orig_01 = kp_0(val_01,:);
    trck_01 = kp_01(val_01,:);
    % Draw flow arrows (color-coded by magnitude)
    flow_m = sqrt(sum((trck_01 - orig_01).^2, 2));
    max_flow = max(flow_m) + 1e-6;
    for ii = 1:size(orig_01,1)
        c_val = flow_m(ii) / max_flow;
        line_color = [c_val, 1-c_val, 0.3];
        plot([orig_01(ii,1), trck_01(ii,1)], [orig_01(ii,2), trck_01(ii,2)], ...
            '-', 'Color', line_color, 'LineWidth', 1.5);
        plot(trck_01(ii,1), trck_01(ii,2), '.', 'Color', line_color, 'MarkerSize', 8);
    end
end
title(sprintf('Flow 0→1: %d tracked (%.0f%%), mean=%.1fpx', ...
    n_t01, ratio_01, mean(flow_mag_01)), 'Color', 'w', 'FontSize', 11);
hold off;

% Panel 2: Flow 0→2
subplot(1,2,2);
imshow(img2_disp); hold on;
if n_t02 > 0
    orig_02 = kp_0(val_02,:);
    trck_02 = kp_02(val_02,:);
    flow_m2 = sqrt(sum((trck_02 - orig_02).^2, 2));
    max_flow2 = max(flow_m2) + 1e-6;
    for ii = 1:size(orig_02,1)
        c_val = flow_m2(ii) / max_flow2;
        line_color = [c_val, 1-c_val, 0.3];
        plot([orig_02(ii,1), trck_02(ii,1)], [orig_02(ii,2), trck_02(ii,2)], ...
            '-', 'Color', line_color, 'LineWidth', 1.5);
        plot(trck_02(ii,1), trck_02(ii,2), '.', 'Color', line_color, 'MarkerSize', 8);
    end
end
title(sprintf('Flow 0→2: %d tracked (%.0f%%), mean=%.1fpx', ...
    n_t02, ratio_02, mean(flow_mag_02)), 'Color', 'w', 'FontSize', 11);
hold off;

sgtitle('Test 2: KLT Optical Flow (green=slow, red=fast)', 'Color', 'w', 'FontSize', 15, 'FontWeight', 'bold');

% ===== FIGURE 3: RANSAC Inliers / Outliers =====
fig3 = figure('Name', 'Real-World VO: RANSAC + Epipolar', ...
    'Position', [100 350 1400 500], 'Color', [0.06 0.06 0.10]);

subplot(1,2,1);
imshow(img0_disp); hold on;
if n_t01 >= 8
    % All tracks (faded red = outliers)
    all_pts_0 = kp_0(val_01,:);
    all_pts_1 = kp_01(val_01,:);

    % Find inlier/outlier mask relative to tracked points
    % p0_in_01 contains the inlier subset
    is_inlier = false(size(all_pts_0, 1), 1);
    for ii = 1:size(p0_in_01, 1)
        dists = sqrt(sum((all_pts_0 - p0_in_01(ii,:)).^2, 2));
        [~, idx] = min(dists);
        if dists(idx) < 0.5
            is_inlier(idx) = true;
        end
    end

    % Draw outliers first (red)
    outlier_idx = find(~is_inlier);
    for ii = outlier_idx'
        plot([all_pts_0(ii,1), all_pts_1(ii,1)], ...
             [all_pts_0(ii,2), all_pts_1(ii,2)], 'r-', 'LineWidth', 1);
        plot(all_pts_0(ii,1), all_pts_0(ii,2), 'rx', 'MarkerSize', 6);
    end
    % Draw inliers (green)
    inlier_idx = find(is_inlier);
    for ii = inlier_idx'
        plot([all_pts_0(ii,1), all_pts_1(ii,1)], ...
             [all_pts_0(ii,2), all_pts_1(ii,2)], 'g-', 'LineWidth', 1.5);
        plot(all_pts_0(ii,1), all_pts_0(ii,2), 'g+', 'MarkerSize', 7, 'LineWidth', 1.5);
    end
end
title(sprintf('RANSAC 0→1: %d inliers (green) / %d outliers (red)', ...
    n_in_01, n_t01 - n_in_01), 'Color', 'w', 'FontSize', 11);
hold off;

subplot(1,2,2);
if r01_ok
    % Visualise epipolar lines on Frame 1
    imshow(img1_disp); hold on;
    plot(p1_in_01(:,1), p1_in_01(:,2), 'g+', 'MarkerSize', 8, 'LineWidth', 1.5);

    % Draw a few epipolar lines
    n_show = min(15, size(p0_in_01,1));
    for ii = 1:n_show
        l = F_01 * [p0_in_01(ii,:)'; 1];  % epipolar line ax+by+c=0
        if abs(l(2)) > 1e-8
            x_line = [1, target_w];
            y_line = -(l(1)*x_line + l(3)) / l(2);
            plot(x_line, y_line, 'c-', 'LineWidth', 0.6);
        end
    end
    title('Epipolar Lines on Frame 1', 'Color', 'w', 'FontSize', 11);
    hold off;
else
    text(0.5, 0.5, 'Not enough data for epipolar lines', ...
        'HorizontalAlignment', 'center', 'Color', 'w', 'FontSize', 14);
end

sgtitle('Test 3: RANSAC Filtering + Epipolar Geometry', 'Color', 'w', 'FontSize', 15, 'FontWeight', 'bold');

% ===== FIGURE 4: 3D Triangulation =====
fig4 = figure('Name', 'Real-World VO: 3D Structure', ...
    'Position', [150 50 1000 700], 'Color', [0.06 0.06 0.10]);

if tri_ok && n_valid_tri > 0
    % 3D scatter of triangulated points
    subplot(2,2,[1,3]);
    scatter3(tri_points(:,1), tri_points(:,2), tri_points(:,3), ...
        40, tri_points(:,3), 'filled');
    hold on;
    % Draw camera positions
    plot3(0, 0, 0, 'g^', 'MarkerSize', 14, 'MarkerFaceColor', 'g', 'LineWidth', 2);
    plot3(best_t(1), best_t(2), best_t(3), 'r^', 'MarkerSize', 14, ...
        'MarkerFaceColor', 'r', 'LineWidth', 2);
    % Draw camera baseline
    plot3([0, best_t(1)], [0, best_t(2)], [0, best_t(3)], 'w-', 'LineWidth', 2);

    colorbar('Color', 'w');
    xlabel('X', 'Color', 'w'); ylabel('Y', 'Color', 'w'); zlabel('Depth (Z)', 'Color', 'w');
    title(sprintf('3D Reconstruction: %d points', n_valid_tri), 'Color', 'w', 'FontSize', 13);
    legend({'3D points', 'Camera 0', 'Camera 1', 'Baseline'}, ...
        'TextColor', 'w', 'Color', [0.12 0.12 0.18], 'Location', 'best');
    set(gca, 'Color', [0.10 0.10 0.15], 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
    grid on; view(45, 30);

    % Top-down view
    subplot(2,2,2);
    scatter(tri_points(:,1), tri_points(:,3), 25, tri_points(:,2), 'filled');
    hold on;
    plot(0, 0, 'g^', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
    plot(best_t(1), best_t(3), 'r^', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    xlabel('X', 'Color', 'w'); ylabel('Depth (Z)', 'Color', 'w');
    title('Top-Down View (X vs Depth)', 'Color', 'w');
    set(gca, 'Color', [0.10 0.10 0.15], 'XColor', 'w', 'YColor', 'w');
    colorbar('Color', 'w'); grid on;

    % Depth histogram
    subplot(2,2,4);
    histogram(tri_points(:,3), 20, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'w');
    xlabel('Depth (camera units)', 'Color', 'w'); ylabel('Count', 'Color', 'w');
    title('Depth Distribution', 'Color', 'w');
    set(gca, 'Color', [0.10 0.10 0.15], 'XColor', 'w', 'YColor', 'w');
    grid on;
else
    text(0.5, 0.5, 'Triangulation not available', ...
        'HorizontalAlignment', 'center', 'Color', 'w', 'FontSize', 16, ...
        'Units', 'normalized');
end

sgtitle('Test 5: 3D Scene Reconstruction from Two Views', 'Color', 'w', 'FontSize', 15, 'FontWeight', 'bold');

fprintf('\n✓ Real-world demonstration complete. 4 figures generated.\n');

%% ========== Helper Functions ==========

function gray = rgb2gray_safe(img)
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
end

function s = pass_fail(b)
    if b, s = '[PASS]'; else, s = '[FAIL]'; end
end

function q = analyze_spatial(kp, W, H)
    % Count features in each quadrant [TL, TR, BL, BR]
    cx = W/2; cy = H/2;
    q = zeros(1,4);
    q(1) = sum(kp(:,1) <= cx & kp(:,2) <= cy);  % Top-left
    q(2) = sum(kp(:,1) >  cx & kp(:,2) <= cy);  % Top-right
    q(3) = sum(kp(:,1) <= cx & kp(:,2) >  cy);  % Bottom-left
    q(4) = sum(kp(:,1) >  cx & kp(:,2) >  cy);  % Bottom-right
end
