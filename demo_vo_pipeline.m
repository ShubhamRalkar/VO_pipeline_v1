%% demo_vo_pipeline.m — Comprehensive VO Pipeline Demonstration
% =========================================================================
%  This script verifies every module in the Visual Odometry system:
%
%   Module 0: Camera Frame Generation     (generate_camera_frame)
%   Module 1: Front-End Features          (detect_orb_features, track_klt_features, ransac_filter)
%   Module 2: Map Initialisation          (initialize_map)
%   Module 3: Back-End Pose Estimation    (run_backend, pnp_solver)
%   Module 4: Full Pipeline               (vo_new_block)
%
%  Run from inside C:\DroneSimulation\Map_initialiser
%  Outputs: Console PASS/FAIL + 4 diagnostic figures
% =========================================================================

clear; clc; clear functions;
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║         VISUAL ODOMETRY PIPELINE DEMONSTRATION            ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

%% ---- Load parameters ----
run('drone_params.m');

% Build parameter structs (used by multiple modules)
cam_p.fx          = cam_fx;
cam_p.fy          = cam_fy;
cam_p.cx          = cam_cx;
cam_p.cy          = cam_cy;
cam_p.width       = frame_width;
cam_p.height      = frame_height;
cam_p.blob_radius = blob_radius;
cam_p.blob_sigma  = blob_sigma;
cam_p.min_depth   = min_depth;
cam_p.max_depth   = max_depth;

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

kf_p.min_features    = kf_min_features;
kf_p.max_translation = kf_max_translation;
kf_p.max_rotation    = kf_max_rotation;

% Test trajectory (circular)
omega = circle_speed / circle_radius;

% Results tracking
results = struct('module', {}, 'test', {}, 'pass', {}, 'detail', {});
add_result = @(m, t, p, d) struct('module', m, 'test', t, 'pass', p, 'detail', d);

%% ================================================================
%  MODULE 0: Camera Frame Generation
% ================================================================
fprintf('\n┌────────────────────────────────────────────┐\n');
fprintf('│  MODULE 0: Camera Frame Generation         │\n');
fprintf('└────────────────────────────────────────────┘\n');

pos_test   = [circle_radius; 0.0; circle_alt];
euler_test = [0; 0; 0];

tic;
frame_0 = generate_camera_frame(pos_test, euler_test, landmarks_3D', cam_p);
t_gen = toc;

% Checks
sz_ok   = isequal(size(frame_0), [frame_height, frame_width]);
type_ok = isa(frame_0, 'uint8');
has_content = max(frame_0(:)) > 50;  % Not blank
brightness_range = double(max(frame_0(:))) - double(min(frame_0(:)));

fprintf('  Frame size:       %dx%d  (expected %dx%d) %s\n', ...
    size(frame_0,2), size(frame_0,1), frame_width, frame_height, bool2str(sz_ok));
fprintf('  Data type:        %s  (expected uint8) %s\n', ...
    class(frame_0), bool2str(type_ok));
fprintf('  Has content:      max=%d min=%d range=%d %s\n', ...
    max(frame_0(:)), min(frame_0(:)), brightness_range, bool2str(has_content));
fprintf('  Generation time:  %.1f ms\n', t_gen*1000);

results(end+1) = add_result('M0', 'Frame size correct',   sz_ok,       sprintf('%dx%d', size(frame_0,2), size(frame_0,1)));
results(end+1) = add_result('M0', 'Frame type uint8',     type_ok,     class(frame_0));
results(end+1) = add_result('M0', 'Frame has content',    has_content, sprintf('range=%d', brightness_range));

% Test 2: Frame changes with different poses
pos_test2   = [circle_radius; 5; circle_alt];
euler_test2 = [0; 0; pi/4];
frame_1 = generate_camera_frame(pos_test2, euler_test2, landmarks_3D', cam_p);
frames_differ = sum(abs(double(frame_0(:)) - double(frame_1(:)))) > 1000;
fprintf('  Frames differ:    %s (different poses produce different frames)\n', bool2str(frames_differ));
results(end+1) = add_result('M0', 'Pose changes frame',   frames_differ, '');

%% ================================================================
%  MODULE 1a: ORB Feature Detection
% ================================================================
fprintf('\n┌────────────────────────────────────────────┐\n');
fprintf('│  MODULE 1a: ORB Feature Detection          │\n');
fprintf('└────────────────────────────────────────────┘\n');

tic;
[kp_0, desc_0, n_det] = detect_orb_features(frame_0, orb_p);
t_orb = toc;

kp_ok   = n_det >= 20;
desc_ok = size(desc_0, 1) == n_det;
kp_bounds = all(kp_0(:,1) >= 1 & kp_0(:,1) <= frame_width & ...
                kp_0(:,2) >= 1 & kp_0(:,2) <= frame_height);

fprintf('  Detected:         %d features %s\n', n_det, bool2str(kp_ok));
fprintf('  Descriptors:      %dx%d %s\n', size(desc_0,1), size(desc_0,2), bool2str(desc_ok));
fprintf('  Bounds check:     all in [1,%d]x[1,%d] %s\n', frame_width, frame_height, bool2str(kp_bounds));
fprintf('  Detection time:   %.1f ms\n', t_orb*1000);

results(end+1) = add_result('M1a', 'ORB detects >= 20',   kp_ok,    sprintf('%d detected', n_det));
results(end+1) = add_result('M1a', 'Descriptor count',    desc_ok,  sprintf('%dx%d', size(desc_0)));
results(end+1) = add_result('M1a', 'Keypoints in bounds', kp_bounds, '');

%% ================================================================
%  MODULE 1b: KLT Tracking
% ================================================================
fprintf('\n┌────────────────────────────────────────────┐\n');
fprintf('│  MODULE 1b: KLT Feature Tracking           │\n');
fprintf('└────────────────────────────────────────────┘\n');

% Generate a second frame with small motion (should track well)
angle_small = omega * 0.5;  % 0.5s of motion
pos_1s   = [circle_radius*cos(angle_small); circle_radius*sin(angle_small); circle_alt];
euler_1s = [0; 0; angle_small];
frame_1s = generate_camera_frame(pos_1s, euler_1s, landmarks_3D', cam_p);

tic;
[kp_tracked, validity, n_tracked] = track_klt_features(frame_0, frame_1s, kp_0, klt_p);
t_klt = toc;

track_ratio = n_tracked / n_det;
track_ok = track_ratio > 0.3;

fprintf('  Tracked:          %d / %d (%.0f%%) %s\n', n_tracked, n_det, track_ratio*100, bool2str(track_ok));
fprintf('  Tracking time:    %.1f ms\n', t_klt*1000);

% Check that tracked points moved (not static)
if n_tracked > 0
    valid_orig = kp_0(validity,:);
    valid_tracked = kp_tracked(validity,:);
    mean_displacement = mean(sqrt(sum((valid_tracked - valid_orig).^2, 2)));
    moved_ok = mean_displacement > 0.5;
    fprintf('  Mean displacement: %.1f px %s\n', mean_displacement, bool2str(moved_ok));
else
    moved_ok = false;
    mean_displacement = 0;
end

results(end+1) = add_result('M1b', 'KLT tracks > 30%%',    track_ok,  sprintf('%.0f%%', track_ratio*100));
results(end+1) = add_result('M1b', 'Tracked pts moved',    moved_ok,  sprintf('%.1f px', mean_displacement));

%% ================================================================
%  MODULE 1c: RANSAC Filtering
% ================================================================
fprintf('\n┌────────────────────────────────────────────┐\n');
fprintf('│  MODULE 1c: RANSAC Outlier Filtering       │\n');
fprintf('└────────────────────────────────────────────┘\n');

tic;
[p0_in, pK_in, F_mat, n_inliers] = ransac_filter(kp_0(validity,:), kp_tracked(validity,:), ran_p);
t_ransac = toc;

ransac_ok = n_inliers >= 8;
F_rank_ok = rank(F_mat) <= 2;  % Fundamental matrix should have rank 2

fprintf('  Inliers:          %d / %d %s\n', n_inliers, n_tracked, bool2str(ransac_ok));
fprintf('  F matrix rank:    %d (expected <=2) %s\n', rank(F_mat), bool2str(F_rank_ok));
fprintf('  RANSAC time:      %.1f ms\n', t_ransac*1000);

results(end+1) = add_result('M1c', 'RANSAC inliers >= 8', ransac_ok,  sprintf('%d inliers', n_inliers));
results(end+1) = add_result('M1c', 'F matrix rank <= 2',  F_rank_ok,  sprintf('rank=%d', rank(F_mat)));

%% ================================================================
%  MODULE 2: Map Initialisation
% ================================================================
fprintf('\n┌────────────────────────────────────────────┐\n');
fprintf('│  MODULE 2: Map Initialisation              │\n');
fprintf('└────────────────────────────────────────────┘\n');

% Find a frame with sufficient baseline
fprintf('  Searching for baseline >= %.1fm...\n', min_baseline);

pos_0_init   = [circle_radius; 0.0; circle_alt];
euler_0_init = [0; 0; 0];
frame_0_init = generate_camera_frame(pos_0_init, euler_0_init, landmarks_3D', cam_p);
[kp_0_init, ~, ~] = detect_orb_features(frame_0_init, orb_p);

baseline_reached = false;
for k = 1:300
    angle_k = omega * k / 30;
    pos_K   = [circle_radius*cos(angle_k); circle_radius*sin(angle_k); circle_alt];
    euler_K = [0; 0; angle_k];
    [ready, dist] = check_baseline(pos_0_init, pos_K, min_baseline);
    if ready
        fprintf('  Baseline reached at frame %d: %.3f m\n', k, dist);
        frame_K = generate_camera_frame(pos_K, euler_K, landmarks_3D', cam_p);
        baseline_reached = true;
        init_frame_idx = k;
        break;
    end
end

baseline_ok = baseline_reached;
results(end+1) = add_result('M2', 'Baseline reached', baseline_ok, sprintf('%.3fm at frame %d', dist, k));

if baseline_reached
    % Front-end for init pair
    [kp2_init, val_init, ~] = track_klt_features(frame_0_init, frame_K, kp_0_init, klt_p);
    [p0_i, pK_i, ~, n_in_i] = ransac_filter(kp_0_init(val_init,:), kp2_init(val_init,:), ran_p);

    tic;
    [lm_cam0, ~, ~, ~, ~, scale, ok_init, n_lm] = initialize_map(...
        frame_0_init, frame_K, p0_i, pK_i, pos_0_init, pos_K, cam_p);
    t_init = toc;

    init_ok    = ok_init;
    enough_lm  = n_lm >= min_init_points;
    scale_ok   = scale > 0.1 && scale < 100;

    fprintf('  Init success:     %s\n', bool2str(init_ok));
    fprintf('  Landmarks:        %d %s\n', n_lm, bool2str(enough_lm));
    fprintf('  Scale:            %.4f %s\n', scale, bool2str(scale_ok));
    fprintf('  Init time:        %.1f ms\n', t_init*1000);

    results(end+1) = add_result('M2', 'Map init succeeds',      init_ok,   '');
    results(end+1) = add_result('M2', 'Enough landmarks',       enough_lm, sprintf('%d >= %d', n_lm, min_init_points));
    results(end+1) = add_result('M2', 'Scale reasonable',       scale_ok,  sprintf('%.4f', scale));
else
    fprintf('  [SKIP] Baseline never reached\n');
    init_ok = false;
    lm_cam0 = []; n_lm = 0; scale = 0;
end

%% ================================================================
%  MODULE 2b: Transform Map to World + Projection Validation
% ================================================================
if init_ok
    fprintf('\n┌────────────────────────────────────────────┐\n');
    fprintf('│  MODULE 2b: Map → World Transform          │\n');
    fprintf('└────────────────────────────────────────────┘\n');

    % Build R_be at frame 0
    roll_0=euler_0_init(1); pitch_0=euler_0_init(2); yaw_0=euler_0_init(3);
    cr0=cos(roll_0); sr0=sin(roll_0); cp0=cos(pitch_0);
    sp0=sin(pitch_0); cy0=cos(yaw_0); sy0=sin(yaw_0);
    R_be_0 = [cy0*cp0,  cy0*sp0*sr0-sy0*cr0,  cy0*sp0*cr0+sy0*sr0;
              sy0*cp0,  sy0*sp0*sr0+cy0*cr0,  sy0*sp0*cr0-cy0*sr0;
              -sp0,     cp0*sr0,               cp0*cr0            ];

    n_valid_lm = double(n_lm);
    map_3d_world = zeros(n_valid_lm, 3);
    for i = 1:n_valid_lm
        p_body = R_cam_body' * lm_cam0(i,:)';
        p_world = R_be_0 * p_body + pos_0_init;
        map_3d_world(i,:) = p_world';
    end

    % Verify: project map into frame_K
    R_ec_K = build_Rec(euler_K, R_cam_body);
    proj_count = 0;
    for m = 1:n_valid_lm
        L_cam = R_ec_K * (map_3d_world(m,:)' - pos_K);
        if L_cam(3) < 0.5 || L_cam(3) > 50, continue; end
        u = cam_fx * L_cam(1)/L_cam(3) + cam_cx;
        v = cam_fy * L_cam(2)/L_cam(3) + cam_cy;
        if u>=1 && u<=frame_width && v>=1 && v<=frame_height
            proj_count = proj_count + 1;
        end
    end

    proj_ok = proj_count >= 10;
    fprintf('  Map points visible in frame_K: %d / %d %s\n', proj_count, n_valid_lm, bool2str(proj_ok));
    fprintf('  Map X range: [%.1f, %.1f]\n', min(map_3d_world(:,1)), max(map_3d_world(:,1)));
    fprintf('  Map Y range: [%.1f, %.1f]\n', min(map_3d_world(:,2)), max(map_3d_world(:,2)));
    fprintf('  Map Z range: [%.1f, %.1f]\n', min(map_3d_world(:,3)), max(map_3d_world(:,3)));

    results(end+1) = add_result('M2b', 'Map projects into keyframe', proj_ok, sprintf('%d visible', proj_count));
end

%% ================================================================
%  MODULE 3: Back-End Tracking (PnP + map maintenance)
% ================================================================
fprintf('\n┌────────────────────────────────────────────┐\n');
fprintf('│  MODULE 3: Back-End Tracking (10 frames)   │\n');
fprintf('└────────────────────────────────────────────┘\n');

if init_ok
    % Set up padded arrays for run_backend (expects 2000-row arrays)
    map_3d_pad   = zeros(2000, 3);
    map_3d_pad(1:n_valid_lm, :) = map_3d_world;
    map_sz = int32(n_valid_lm);

    % Project map into keyframe K to get initial kp_kf
    kp_kf_pad = project_map_into_frame(map_3d_pad, n_valid_lm, pos_K, euler_K, ...
        R_cam_body, cam_fx, cam_fy, cam_cx, cam_cy, frame_width, frame_height, min_depth, max_depth);

    kp_kf_init = detect_orb_features(frame_K, orb_p);
    pos_kf     = pos_K;
    euler_kf   = euler_K;
    last_pos   = pos_K;
    last_euler = euler_K;

    N_backend_frames = 10;
    errors     = zeros(N_backend_frames, 1);
    valid_mask = false(N_backend_frames, 1);
    est_positions = zeros(N_backend_frames, 3);
    true_positions = zeros(N_backend_frames, 3);

    fprintf('  %-5s  %-28s  %-28s  %-8s  %-4s\n', ...
        'Frame', 'True Position', 'Estimated Position', 'Error', 'OK');
    fprintf('  %s\n', repmat('-', 1, 78));

    frame_prev_be = frame_K;

    for k = 1:N_backend_frames
        t_now  = init_frame_idx/30 + k/30;
        angle_be = omega * t_now;

        pos_curr   = [circle_radius*cos(angle_be); circle_radius*sin(angle_be); circle_alt];
        euler_curr = [0; 0; angle_be];

        frame_curr = generate_camera_frame(pos_curr, euler_curr, landmarks_3D', cam_p);

        [pos_vo, euler_vo, vo_valid, map_3d_pad, map_sz, kp_kf_pad] = run_backend(...
            frame_prev_be, frame_curr, ...
            kp_kf_pad, pos_kf, euler_kf, ...
            pos_curr, euler_curr, ...
            map_3d_pad, map_sz, ...
            cam_p, R_cam_body, ...
            orb_p, klt_p, ran_p, kf_p, ...
            last_pos, last_euler);

        err = norm(pos_vo - pos_curr);
        errors(k)     = err;
        valid_mask(k) = vo_valid;
        est_positions(k,:) = pos_vo';
        true_positions(k,:) = pos_curr';

        fprintf('  %-5d  [%7.2f,%7.2f,%7.2f]  [%7.2f,%7.2f,%7.2f]  %6.3fm  %s\n', ...
            k, pos_curr(1), pos_curr(2), pos_curr(3), ...
            pos_vo(1), pos_vo(2), pos_vo(3), err, bool2str(vo_valid));

        if vo_valid
            last_pos   = pos_vo;
            last_euler = euler_vo;

            d_pos = norm(pos_curr - pos_kf);
            if d_pos > kf_p.max_translation
                kp_kf_pad = project_map_into_frame(map_3d_pad, double(map_sz), ...
                    pos_curr, euler_curr, R_cam_body, cam_fx, cam_fy, cam_cx, cam_cy, ...
                    frame_width, frame_height, min_depth, max_depth);
                pos_kf   = pos_curr;
                euler_kf = euler_curr;
            end
        end

        frame_prev_be = frame_curr;
    end

    n_valid = sum(valid_mask);
    be_valid_ok = n_valid >= 5;
    if any(valid_mask)
        mean_err = mean(errors(valid_mask));
        be_err_ok = mean_err < 3.0;
    else
        mean_err = Inf;
        be_err_ok = false;
    end

    fprintf('\n  Valid frames: %d/10 %s\n', n_valid, bool2str(be_valid_ok));
    fprintf('  Mean error:   %.3f m %s\n', mean_err, bool2str(be_err_ok));
    fprintf('  Map size:     %d\n', map_sz);

    results(end+1) = add_result('M3', 'Backend valid >= 5',     be_valid_ok, sprintf('%d/10', n_valid));
    results(end+1) = add_result('M3', 'Mean error < 3m',        be_err_ok,   sprintf('%.3fm', mean_err));
else
    fprintf('  [SKIP] Map init failed, cannot test backend\n');
end

%% ================================================================
%  MODULE 4: Helper Function Tests
% ================================================================
fprintf('\n┌────────────────────────────────────────────┐\n');
fprintf('│  MODULE 4: Helper Function Tests           │\n');
fprintf('└────────────────────────────────────────────┘\n');

% check_baseline
[ready_no, dist_no] = check_baseline([0;0;0], [0;0;0], 1.0);
[ready_yes, dist_yes] = check_baseline([0;0;0], [2;0;0], 1.0);
bl_ok = ~ready_no && ready_yes && abs(dist_yes - 2.0) < 0.01;
fprintf('  check_baseline:   %s (near=%.2fm/ready=%d, far=%.2fm/ready=%d)\n', ...
    bool2str(bl_ok), dist_no, ready_no, dist_yes, ready_yes);
results(end+1) = add_result('M4', 'check_baseline', bl_ok, '');

% build_Rec
R_test = build_Rec([0;0;0], R_cam_body);
rec_ok = norm(R_test - R_cam_body) < 1e-10;  % At zero euler, R_ec = R_cam_body * I' = R_cam_body
fprintf('  build_Rec:        %s (identity euler → R_ec = R_cam_body)\n', bool2str(rec_ok));
results(end+1) = add_result('M4', 'build_Rec zero euler', rec_ok, '');

% compute_Rbe
R_be_test = compute_Rbe([0;0;0]);
rbe_ok = norm(R_be_test - eye(3)) < 1e-10;
fprintf('  compute_Rbe:      %s (identity euler → R_be = I)\n', bool2str(rbe_ok));
results(end+1) = add_result('M4', 'compute_Rbe zero euler', rbe_ok, '');

% project_map_into_frame
if init_ok
    pts2d_test = project_map_into_frame(map_3d_pad, n_valid_lm, pos_K, euler_K, ...
        R_cam_body, cam_fx, cam_fy, cam_cx, cam_cy, frame_width, frame_height, min_depth, max_depth);
    n_projected = sum(any(pts2d_test ~= 0, 2));
    proj_fn_ok = n_projected > 0;
    fprintf('  project_map:      %s (%d points project into frame)\n', bool2str(proj_fn_ok), n_projected);
    results(end+1) = add_result('M4', 'project_map_into_frame', proj_fn_ok, sprintf('%d projected', n_projected));
end

%% ================================================================
%  RESULTS SUMMARY
% ================================================================
fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║                    RESULTS SUMMARY                        ║\n');
fprintf('╠════════════════════════════════════════════════════════════╣\n');

n_pass = 0;
n_fail = 0;
for i = 1:length(results)
    if results(i).pass
        status = '✓ PASS';
        n_pass = n_pass + 1;
    else
        status = '✗ FAIL';
        n_fail = n_fail + 1;
    end
    detail_str = results(i).detail;
    if ~isempty(detail_str)
        detail_str = [' (' detail_str ')'];
    end
    fprintf('║  %-5s %-35s %s%s\n', results(i).module, results(i).test, status, detail_str);
end

fprintf('╠════════════════════════════════════════════════════════════╣\n');
fprintf('║  TOTAL: %d passed, %d failed out of %d tests', n_pass, n_fail, n_pass+n_fail);
fprintf('%s║\n', repmat(' ', 1, max(0, 20 - length(sprintf('%d passed, %d failed out of %d tests', n_pass, n_fail, n_pass+n_fail)))));
if n_fail == 0
    fprintf('║  STATUS: ALL TESTS PASSED ✓                              ║\n');
else
    fprintf('║  STATUS: %d TEST(S) FAILED ✗                             ║\n', n_fail);
end
fprintf('╚════════════════════════════════════════════════════════════╝\n');

%% ================================================================
%  VISUALISATION
% ================================================================
fprintf('\nGenerating visualisation figures...\n');

% ---- Figure 1: Camera Frames + Feature Detection ----
fig1 = figure('Name', 'VO Demo: Camera & Features', ...
    'Position', [50 500 1200 500], 'Color', [0.08 0.08 0.12]);

subplot(1,3,1);
imshow(frame_0); hold on;
plot(kp_0(:,1), kp_0(:,2), 'g+', 'MarkerSize', 5, 'LineWidth', 1);
title(sprintf('Frame 0: %d ORB features', n_det), 'Color', 'w', 'FontSize', 12);
subtitle(sprintf('pos=[%.1f, %.1f, %.1f]', pos_test(1), pos_test(2), pos_test(3)), 'Color', [0.7 0.7 0.7]);

subplot(1,3,2);
imshow(frame_1s); hold on;
if n_tracked > 0
    plot(kp_tracked(validity,1), kp_tracked(validity,2), 'c+', 'MarkerSize', 5, 'LineWidth', 1);
    % Draw motion vectors
    for ii = 1:min(50, size(valid_orig,1))
        plot([valid_orig(ii,1), valid_tracked(ii,1)], ...
             [valid_orig(ii,2), valid_tracked(ii,2)], 'y-', 'LineWidth', 0.8);
    end
end
title(sprintf('Frame 1: %d KLT tracked', n_tracked), 'Color', 'w', 'FontSize', 12);
subtitle(sprintf('%.1f px mean displacement', mean_displacement), 'Color', [0.7 0.7 0.7]);

subplot(1,3,3);
imshow(frame_K); hold on;
plot(pK_i(:,1), pK_i(:,2), 'r+', 'MarkerSize', 5, 'LineWidth', 1);
title(sprintf('Frame K: %d RANSAC inliers', n_in_i), 'Color', 'w', 'FontSize', 12);
subtitle(sprintf('baseline=%.2fm', dist), 'Color', [0.7 0.7 0.7]);

sgtitle('Module 0+1: Camera Frames, ORB, KLT, RANSAC', 'Color', 'w', 'FontSize', 14);

% ---- Figure 2: Map Initialisation ----
if init_ok
    fig2 = figure('Name', 'VO Demo: Map Initialisation', ...
        'Position', [50 50 1200 500], 'Color', [0.08 0.08 0.12]);

    subplot(1,3,1);
    imshow(frame_0_init); hold on;
    plot(p0_i(:,1), p0_i(:,2), 'g+', 'MarkerSize', 6, 'LineWidth', 1.5);
    title(sprintf('Frame 0 — %d inlier pts', size(p0_i,1)), 'Color', 'w');

    subplot(1,3,2);
    imshow(frame_K); hold on;
    plot(pK_i(:,1), pK_i(:,2), 'b+', 'MarkerSize', 6, 'LineWidth', 1.5);
    title(sprintf('Frame K — %d inlier pts', size(pK_i,1)), 'Color', 'w');

    subplot(1,3,3);
    if n_valid_lm > 0
        scatter3(map_3d_world(:,1), map_3d_world(:,2), map_3d_world(:,3), ...
            20, map_3d_world(:,3), 'filled');
        hold on;
        % Plot known landmarks (faded)
        scatter3(landmarks_3D(:,1), landmarks_3D(:,2), landmarks_3D(:,3), ...
            3, [0.3 0.3 0.3], '.', 'MarkerFaceAlpha', 0.2);
        colorbar('Color', 'w');
        xlabel('X (N)', 'Color', 'w'); ylabel('Y (E)', 'Color', 'w'); zlabel('Z (D)', 'Color', 'w');
        title(sprintf('Triangulated: %d landmarks, scale=%.3f', n_valid_lm, scale), 'Color', 'w');
        set(gca, 'Color', [0.12 0.12 0.18], 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
        grid on; view(45, 30);
    end

    sgtitle('Module 2: Map Initialisation', 'Color', 'w', 'FontSize', 14);
end

% ---- Figure 3: Back-End Tracking Results ----
if init_ok
    fig3 = figure('Name', 'VO Demo: Backend Tracking', ...
        'Position', [100 300 1200 500], 'Color', [0.08 0.08 0.12]);

    subplot(1,3,1);
    valid_idx = find(valid_mask);
    if ~isempty(valid_idx)
        bar(1:N_backend_frames, errors, 'FaceColor', [0.2 0.6 0.9]);
        hold on;
        bar(find(~valid_mask), errors(~valid_mask), 'FaceColor', [0.8 0.2 0.2]);
        yline(1.0, 'g--', '1m', 'LineWidth', 1.5, 'LabelColor', 'g');
        yline(3.0, 'r--', '3m', 'LineWidth', 1.5, 'LabelColor', 'r');
    end
    xlabel('Frame', 'Color', 'w'); ylabel('Error (m)', 'Color', 'w');
    title(sprintf('Position Error (mean=%.3fm)', mean_err), 'Color', 'w');
    set(gca, 'Color', [0.12 0.12 0.18], 'XColor', 'w', 'YColor', 'w');
    grid on;

    subplot(1,3,2);
    % True vs estimated trajectory (top-down)
    plot(true_positions(:,1), true_positions(:,2), 'g-o', 'LineWidth', 2, ...
        'MarkerSize', 6, 'MarkerFaceColor', 'g');
    hold on;
    if any(valid_mask)
        plot(est_positions(valid_mask,1), est_positions(valid_mask,2), 'c-s', ...
            'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'c');
    end
    % Draw circular path
    th_circ = linspace(0, 2*pi, 200);
    plot(circle_radius*cos(th_circ), circle_radius*sin(th_circ), 'w--', ...
        'LineWidth', 0.8);
    legend('True', 'Estimated', 'Path', 'TextColor', 'w', 'Color', [0.15 0.15 0.2]);
    xlabel('X (N)', 'Color', 'w'); ylabel('Y (E)', 'Color', 'w');
    title('Top-Down Trajectory', 'Color', 'w');
    set(gca, 'Color', [0.12 0.12 0.18], 'XColor', 'w', 'YColor', 'w');
    axis equal; grid on;

    subplot(1,3,3);
    % 3D view with map
    scatter3(map_3d_pad(1:double(map_sz),1), map_3d_pad(1:double(map_sz),2), ...
        map_3d_pad(1:double(map_sz),3), 8, [0.3 0.5 0.8], 'filled', ...
        'MarkerFaceAlpha', 0.4);
    hold on;
    plot3(true_positions(:,1), true_positions(:,2), true_positions(:,3), ...
        'g-o', 'LineWidth', 2, 'MarkerSize', 5);
    if any(valid_mask)
        plot3(est_positions(valid_mask,1), est_positions(valid_mask,2), ...
            est_positions(valid_mask,3), 'c-s', 'LineWidth', 2, 'MarkerSize', 5);
    end
    xlabel('X', 'Color', 'w'); ylabel('Y', 'Color', 'w'); zlabel('Z', 'Color', 'w');
    title('3D Map + Trajectory', 'Color', 'w');
    legend('Map', 'True', 'Est.', 'TextColor', 'w', 'Color', [0.15 0.15 0.2]);
    set(gca, 'Color', [0.12 0.12 0.18], 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
    grid on; view(45, 30);

    sgtitle('Module 3: Back-End Tracking', 'Color', 'w', 'FontSize', 14);
end

% ---- Figure 4: 3D Environment Overview ----
fig4 = figure('Name', 'VO Demo: 3D City Environment', ...
    'Position', [150 150 900 700], 'Color', [0.05 0.05 0.08]);

% Plot landmarks colored by type
scatter3(landmarks_3D(:,1), landmarks_3D(:,2), landmarks_3D(:,3), ...
    5, -landmarks_3D(:,3), 'filled');
hold on;

% Plot circular flight path
th_path = linspace(0, 2*pi, 200);
plot3(circle_radius*cos(th_path), circle_radius*sin(th_path), ...
    circle_alt*ones(size(th_path)), 'c-', 'LineWidth', 2.5);

% Plot buildings as wireframes
for b = 1:size(buildings_def, 1)
    bx = buildings_def(b,1); by = buildings_def(b,2);
    bw = buildings_def(b,3); bd = buildings_def(b,4);
    bh = buildings_def(b,5);

    verts = [bx-bw/2, by-bd/2, 0;
             bx+bw/2, by-bd/2, 0;
             bx+bw/2, by+bd/2, 0;
             bx-bw/2, by+bd/2, 0;
             bx-bw/2, by-bd/2, -bh;
             bx+bw/2, by-bd/2, -bh;
             bx+bw/2, by+bd/2, -bh;
             bx-bw/2, by+bd/2, -bh];
    faces = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 5 6 7 8];
    patch('Vertices', verts, 'Faces', faces, ...
        'FaceColor', [0.2 0.25 0.4], 'FaceAlpha', 0.15, ...
        'EdgeColor', [0.4 0.45 0.6], 'LineWidth', 0.5);
end

colorbar('Color', 'w');
xlabel('X North [m]', 'Color', 'w');
ylabel('Y East [m]', 'Color', 'w');
zlabel('Z Down [m]', 'Color', 'w');
title(sprintf('3D City Environment — %d landmarks', num_landmarks), 'Color', 'w', 'FontSize', 14);
set(gca, 'Color', [0.08 0.08 0.12], 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
grid on; axis equal; view(35, 30);
legend('Landmarks', 'Flight path', 'Buildings', 'TextColor', 'w', 'Color', [0.12 0.12 0.18]);

fprintf('\n✓ Demonstration complete. 4 figures generated.\n');

%% ---- Helper function ----
function s = bool2str(b)
    if b
        s = '[PASS]';
    else
        s = '[FAIL]';
    end
end
