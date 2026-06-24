% test_map_initializer.m
clear; clc; clear functions;
run('drone_params.m');

% ---- Build parameter structs ----
cam_params.fx=cam_fx; cam_params.fy=cam_fy;
cam_params.cx=cam_cx; cam_params.cy=cam_cy;
cam_params.width=frame_width; cam_params.height=frame_height;
cam_params.blob_radius=blob_radius; cam_params.blob_sigma=blob_sigma;
cam_params.min_depth=min_depth; cam_params.max_depth=max_depth;

orb_params.num_points=orb_num_points; orb_params.num_levels=orb_num_levels;
orb_params.scale_factor=orb_scale_factor;

klt_params.block_size=klt_block_size; klt_params.max_iterations=klt_max_iterations;
klt_params.num_levels=klt_num_levels; klt_params.max_error=klt_max_error;

ransac_p.max_distance=ransac_max_distance; ransac_p.confidence=ransac_confidence;
ransac_p.max_iterations=ransac_max_iterations;

%-- Frame 0 at start of circular path --------------------------------
pos_0   = [circle_radius; 0.0; circle_alt];
euler_0 = [0; 0; 0];
frame_0 = generate_camera_frame(pos_0, euler_0, landmarks_3D', cam_params);
[kp_0, ~, ~] = detect_orb_features(frame_0, orb_params);

fprintf('=== Map Initializer Test ===\n');
fprintf('Waiting for %.1fm baseline...\n\n', min_baseline);

%-- Simulate drone moving until baseline threshold met ---------------
omega = circle_speed / circle_radius;
baseline_reached = false;
pos_K = pos_0; euler_K = euler_0; frame_K = frame_0;

for k = 1:300
    t     = k / 30;
    angle = omega * t;
    pos_K   = [circle_radius*cos(angle); circle_radius*sin(angle); circle_alt];
    euler_K = [0; 0; angle];

    [ready, dist] = check_baseline(pos_0, pos_K, min_baseline);
    if ready
        fprintf('Baseline reached at frame %d: %.3fm\n\n', k, dist);
        frame_K = generate_camera_frame(pos_K, euler_K, landmarks_3D', cam_params);
        baseline_reached = true;
        break;
    end
end

if ~baseline_reached
    error('Baseline never reached — check circle_speed or min_baseline');
end

%-- Module 1 front-end between frame_0 and frame_K ------------------
fprintf('Running front-end...\n');
[kp2, validity, ~]    = track_klt_features(frame_0, frame_K, kp_0, klt_params);
[p0, pK, ~, n_in]     = ransac_filter(kp_0(validity,:), kp2(validity,:), ransac_p);
fprintf('Front-end inliers: %d\n\n', n_in);

%-- Map Initializer --------------------------------------------------
fprintf('Running map initializer...\n');
[lm_cam0, poses, scale, success] = initialize_map(...
    frame_0, frame_K, p0, pK, pos_0, pos_K, cam_params);

%-- Report -----------------------------------------------------------
fprintf('\n=====================================\n');
fprintf('Map Initializer Results\n');
fprintf('-------------------------------------\n');
fprintf('Success:           %s\n',   string(success));
fprintf('Landmarks:         %d\n',   size(lm_cam0,1));
fprintf('Metric scale:      %.4f\n', scale);
if ~isempty(poses)
    fprintf('Pose 2 t_norm:     %.4f m\n', norm(poses(2).t));
end
fprintf('=====================================\n');

%-- Validation -------------------------------------------------------
fprintf('\nValidation:\n');
if success
    fprintf('  [PASS] Initialisation succeeded\n');
else
    fprintf('  [FAIL] Initialisation failed\n'); return;
end
if size(lm_cam0,1) >= min_init_points
    fprintf('  [PASS] Landmarks >= %d (%d)\n', min_init_points, size(lm_cam0,1));
else
    fprintf('  [FAIL] Too few landmarks: %d\n', size(lm_cam0,1));
end
if scale > 0.1 && scale < 100
    fprintf('  [PASS] Scale reasonable (%.4f)\n', scale);
else
    fprintf('  [FAIL] Scale out of range: %.4f\n', scale);
end

%-- Visualise --------------------------------------------------------
figure('Name','Map Initializer Test','Position',[100 100 1100 450]);

subplot(1,3,1);
imshow(frame_0); hold on;
plot(p0(:,1), p0(:,2), 'g+', 'MarkerSize',6,'LineWidth',1.5);
title(sprintf('Frame 0 — %d pts', size(p0,1)));
subtitle(sprintf('pos=[%.1f, %.1f, %.1f]', pos_0(1),pos_0(2),pos_0(3)));
hold off;

subplot(1,3,2);
imshow(frame_K); hold on;
plot(pK(:,1), pK(:,2), 'b+', 'MarkerSize',6,'LineWidth',1.5);
title(sprintf('Frame K — %d pts', size(pK,1)));
subtitle(sprintf('pos=[%.1f, %.1f, %.1f]', pos_K(1),pos_K(2),pos_K(3)));
hold off;

subplot(1,3,3);
if size(lm_cam0,1) > 0
    scatter3(lm_cam0(:,1), lm_cam0(:,2), lm_cam0(:,3), ...
        25, lm_cam0(:,3), 'filled');
    colorbar;
    xlabel('X_{cam}'); ylabel('Y_{cam}'); zlabel('Z (depth)');
    title(sprintf('Triangulated map: %d landmarks', size(lm_cam0,1)));
    subtitle(sprintf('Scale = %.3f m/unit', scale));
    grid on; axis equal; view(45,30);
end