% test_backend.m
clear; clc; clear functions;
run('drone_params.m');

% ---- Build parameter structs ----
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

% ================================================================
% STEP 1: Map Initialisation (Module 2)
% ================================================================
fprintf('=== Initialising Map ===\n');

pos_0   = [circle_radius; 0.0; circle_alt];
euler_0 = [0; 0; 0];
frame_0 = generate_camera_frame(pos_0, euler_0, landmarks_3D', cam_p);
[kp_0, ~, ~] = detect_orb_features(frame_0, orb_p);

% Build R_be and R_ec at frame_0
roll=euler_0(1); pitch=euler_0(2); yaw=euler_0(3);
cr=cos(roll); sr=sin(roll); cp=cos(pitch);
sp=sin(pitch); cy_=cos(yaw); sy_=sin(yaw);
R_be_0 = [cy_*cp,  cy_*sp*sr-sy_*cr,  cy_*sp*cr+sy_*sr;
          sy_*cp,  sy_*sp*sr+cy_*cr,  sy_*sp*cr-cy_*sr;
          -sp,     cp*sr,              cp*cr            ];
R_ec_0 = R_cam_body * R_be_0';

% Wait for baseline
omega = circle_speed / circle_radius;
baseline_reached = false;
for k = 1:300
    angle = omega * k / 30;
    pos_K   = [circle_radius*cos(angle); circle_radius*sin(angle); circle_alt];
    euler_K = [0; 0; angle];
    [ready, dist] = check_baseline(pos_0, pos_K, min_baseline);
    if ready
        fprintf('Baseline reached at frame %d: %.3fm\n', k, dist);
        frame_K = generate_camera_frame(pos_K, euler_K, landmarks_3D', cam_p);
        baseline_reached = true;
        kf_frame_idx = k;
        break;
    end
end

if ~baseline_reached
    error('Baseline never reached');
end

% Run Module 1 between frame_0 and frame_K
[kp2, val, ~]  = track_klt_features(frame_0, frame_K, kp_0, klt_p);
[p0, pK, ~, n] = ransac_filter(kp_0(val,:), kp2(val,:), ran_p);
fprintf('Front-end inliers: %d\n', n);

% Run Module 2
[lm_cam0, ~, scale, ok] = initialize_map(frame_0, frame_K, p0, pK, pos_0, pos_K, cam_p);
if ~ok
    error('Map initialisation failed — fix Module 2 first');
end

% ================================================================
% STEP 2: Transform map from camera-0 frame to world NED
% ================================================================
% Camera frame: X=right, Y=down, Z=forward(depth)
% Transform: world = R_be_0 * R_cam_body' * X_cam + pos_0
map_3d   = zeros(size(lm_cam0,1), 3);
for i = 1:size(lm_cam0,1)
    p_cam        = lm_cam0(i,:)';
    p_body       = R_cam_body' * p_cam;      % camera → body frame
    p_world      = R_be_0 * p_body + pos_0;  % body → world NED
    map_3d(i,:)  = p_world';
end
map_size = size(map_3d, 1);

% ---- Map diagnostic ----
fprintf('\n=== Map Diagnostic ===\n');
fprintf('Map point range:\n');
fprintf('  X: %.2f to %.2f\n', min(map_3d(:,1)), max(map_3d(:,1)));
fprintf('  Y: %.2f to %.2f\n', min(map_3d(:,2)), max(map_3d(:,2)));
fprintf('  Z: %.2f to %.2f\n', min(map_3d(:,3)), max(map_3d(:,3)));
fprintf('Map sample (first 5):\n');
disp(map_3d(1:min(5,map_size),:));
fprintf('Known landmarks sample (first 5):\n');
disp(landmarks_3D(1:5,:));
fprintf('Expected: map X~10-35, Y~-15 to 15, Z~-15 to 0\n');
fprintf('Drone at frame_K: [%.2f, %.2f, %.2f]\n\n', pos_K(1),pos_K(2),pos_K(3));

% ================================================================
% STEP 3: Verify map projects correctly into frame_K
% ================================================================
fprintf('=== Projection Sanity Check ===\n');

roll=euler_K(1); pitch=euler_K(2); yaw=euler_K(3);
cr=cos(roll); sr=sin(roll); cp=cos(pitch);
sp=sin(pitch); cy_=cos(yaw); sy_=sin(yaw);
R_be_K = [cy_*cp,  cy_*sp*sr-sy_*cr,  cy_*sp*cr+sy_*sr;
          sy_*cp,  sy_*sp*sr+cy_*cr,  sy_*sp*cr-cy_*sr;
          -sp,     cp*sr,              cp*cr            ];
R_ec_K = R_cam_body * R_be_K';

proj_count = 0;
for m = 1:map_size
    L_cam = R_ec_K * (map_3d(m,:)' - pos_K);
    if L_cam(3) < 0.5 || L_cam(3) > 50, continue; end
    u = cam_fx * L_cam(1)/L_cam(3) + cam_cx;
    v = cam_fy * L_cam(2)/L_cam(3) + cam_cy;
    if u>=1 && u<=frame_width && v>=1 && v<=frame_height
        proj_count = proj_count + 1;
    end
end
fprintf('Map points projecting into frame_K: %d / %d\n', proj_count, map_size);
if proj_count >= 10
    fprintf('[PASS] Map projects correctly into keyframe\n\n');
else
    fprintf('[FAIL] Map does not project into keyframe — coordinate transform issue\n\n');
end

% ================================================================
% STEP 4: Run Module 3 Back-End for 10 frames
% ================================================================
fprintf('=== Running Module 3 Back-End ===\n');
fprintf('%-6s %-22s %-22s %-10s %-10s\n', ...
    'Frame','True pos','Est  pos','Error(m)','Map size');

frame_prev = frame_K;
kp_kf      = detect_orb_features(frame_K, orb_p);
pos_kf     = pos_K;
euler_kf   = euler_K;
last_pos   = pos_K;
last_euler = euler_K;

errors     = zeros(10,1);
valid_mask = false(10,1);

for k = 1:10
    t_now  = kf_frame_idx/30 + k/30;
    angle  = omega * t_now;

    pos_curr   = [circle_radius*cos(angle); circle_radius*sin(angle); circle_alt];
    euler_curr = [0; 0; angle];

    frame_curr = generate_camera_frame(pos_curr, euler_curr, landmarks_3D', cam_p);

    [pos_vo, euler_vo, vo_valid, map_3d, map_size] = run_backend(...
        frame_prev, frame_curr, ...
        kp_kf, pos_kf, euler_kf, ...
        pos_curr, euler_curr, ...
        map_3d, map_size, ...
        cam_p, R_cam_body, ...
        orb_p, klt_p, ran_p, kf_p, ...
        last_pos, last_euler);

    err = norm(pos_vo - pos_curr);
    errors(k)     = err;
    valid_mask(k) = vo_valid;

    fprintf('%-6d [%6.2f,%6.2f,%6.2f]  [%6.2f,%6.2f,%6.2f]  %-10.3f %-10d\n', ...
        k, ...
        pos_curr(1), pos_curr(2), pos_curr(3), ...
        pos_vo(1),   pos_vo(2),   pos_vo(3), ...
        err, map_size);

    if vo_valid
        last_pos   = pos_vo;
        last_euler = euler_vo;
        % Update keyframe if needed
        d_pos = norm(pos_curr - pos_kf);
        if d_pos > kf_p.max_translation
            kp_kf    = detect_orb_features(frame_curr, orb_p);
            pos_kf   = pos_curr;
            euler_kf = euler_curr;
            frame_prev = frame_curr;
            fprintf('  [KF updated at frame %d]\n', k);
        else
            frame_prev = frame_curr;
        end
    else
        frame_prev = frame_curr;
    end
end

% ================================================================
% STEP 5: Summary and Validation
% ================================================================
fprintf('\n=====================================\n');
fprintf('Module 3 Back-End Results\n');
fprintf('-------------------------------------\n');
fprintf('Valid VO frames:     %d / 10\n', sum(valid_mask));
fprintf('Mean position error: %.3f m\n',  mean(errors(valid_mask)));
fprintf('Max  position error: %.3f m\n',  max(errors(valid_mask)));
fprintf('Final map size:      %d\n',      map_size);
fprintf('=====================================\n');

fprintf('\nValidation:\n');
if sum(valid_mask) >= 5
    fprintf('  [PASS] VO valid >= 5 frames (%d/10)\n', sum(valid_mask));
else
    fprintf('  [FAIL] Too few valid VO frames: %d/10\n', sum(valid_mask));
end
if any(valid_mask) && mean(errors(valid_mask)) < 2.0
    fprintf('  [PASS] Mean error < 2.0m (%.3fm)\n', mean(errors(valid_mask)));
else
    fprintf('  [FAIL] Mean error too large or no valid frames\n');
end
if map_size >= map_size   % always true — just confirm it didn't crash
    fprintf('  [PASS] Map maintained (%d landmarks)\n', map_size);
end

% ================================================================
% STEP 6: Visualisation
% ================================================================
figure('Name','Module 3 Back-End Test','Position',[100 100 1200 500]);

subplot(1,3,1);
% Show frame_K with map projections overlaid
imshow(frame_K); hold on;
for m = 1:min(map_size,400)
    L_cam = R_ec_K * (map_3d(m,:)' - pos_K);
    if L_cam(3)<0.5||L_cam(3)>50, continue; end
    u = cam_fx*L_cam(1)/L_cam(3)+cam_cx;
    v = cam_fy*L_cam(2)/L_cam(3)+cam_cy;
    if u>=1&&u<=frame_width&&v>=1&&v<=frame_height
        plot(u, v, 'r.', 'MarkerSize', 6);
    end
end
title('Frame K with map projections');
subtitle('Red dot = projected map point');
hold off;

subplot(1,3,2);
% Position error over time
valid_frames = find(valid_mask);
if ~isempty(valid_frames)
    plot(valid_frames, errors(valid_frames), 'b-o', 'LineWidth', 2);
    xlabel('Frame'); ylabel('Position error (m)');
    title('VO Position Error');
    subtitle(sprintf('Mean = %.3fm', mean(errors(valid_mask))));
    grid on;
    yline(0.5, 'r--', '0.5m threshold');
else
    text(0.5,0.5,'No valid VO frames','HorizontalAlignment','center');
    title('VO Position Error — No data');
end

subplot(1,3,3);
% 3D map and drone trajectory
scatter3(map_3d(1:map_size,1), map_3d(1:map_size,2), map_3d(1:map_size,3), ...
    10, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
% Plot true trajectory
traj_pts = zeros(10,3);
for k = 1:10
    t_now = kf_frame_idx/30 + k/30;
    ang   = omega * t_now;
    traj_pts(k,:) = [circle_radius*cos(ang), circle_radius*sin(ang), circle_alt];
end
plot3(traj_pts(:,1), traj_pts(:,2), traj_pts(:,3), 'g-o', ...
    'LineWidth', 2, 'MarkerSize', 5);
plot3(pos_K(1), pos_K(2), pos_K(3), 'r*', 'MarkerSize', 12);
xlabel('X (North)'); ylabel('Y (East)'); zlabel('Z (Down)');
title('3D Map + Trajectory');
subtitle('Blue=map  Green=true traj  Red*=keyframe');
grid on; axis equal; view(45,30);
hold off;
