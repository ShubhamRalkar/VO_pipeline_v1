function [pos_vo, euler_vo, vo_valid, num_vis] = vo_new_block(...
           pos_true, euler_true, ...
           cam_fy, cam_cx, cam_cy, ...
           cam_width, cam_height, ...
           blob_radius, blob_sigma, ...
           min_depth, max_depth, ...
           orb_num_points, orb_num_levels, orb_scale_factor, ...
           klt_block_size_r, klt_block_size_c, ...
           klt_max_iter, klt_num_levels, klt_max_error, ...
           ransac_max_dist, ransac_conf, ransac_max_iter, ...
           kf_min_feat, kf_max_trans, kf_max_rot, ...
           min_baseline_m, vo_hz, Ts, ...
           landmarks_3D_in, num_landmarks, R_cam_body, cam_fx)
%VO_NEW_BLOCK - Full visual odometry pipeline entry for Simulink
%
% Input arguments:
% pos_true, euler_true - true vehicle pose (used for frame generation)
% ... (see function signature for camera, KLT/ORB, RANSAC, KF params)
%
% Output arguments:
% pos_vo, euler_vo - estimated pose from VO
% vo_valid         - validity flag for VO estimate
% num_vis          - number of visible landmarks
%#codegen
% VO_NEW_BLOCK  Full VO pipeline (Module 1+2+3) for Simulink
% Inputs:  true pos/euler from 6-DOF block (frame generation only)
% Outputs: ESTIMATED pos/euler from VO pipeline feeds EKF
%
% kp_kf stores MAP PROJECTIONS into keyframe (not ORB keypoints)
% guarantees correct 2D-3D correspondence via KLT tracking

persistent counter initialized init_checking ...
           frame_prev kp_kf pos_kf euler_kf frame_kf ...
           map_3d map_size ...
           last_pos last_euler ...
           pos_0 euler_0 frame_0 kp_0;

% Persistent state holds previous frames, keyframe map, and flags
if isempty(counter)
    counter       = int32(0);
    initialized   = false;
    init_checking = false;
    frame_prev    = zeros(480, 640, 'uint8');
    frame_kf      = zeros(480, 640, 'uint8');
    frame_0       = zeros(480, 640, 'uint8');
    kp_kf         = zeros(2000, 2);
    kp_0          = zeros(500,  2);
    pos_kf        = zeros(3, 1);
    euler_kf      = zeros(3, 1);
    map_3d        = zeros(2000, 3);
    map_size      = int32(0);
    last_pos      = zeros(3, 1);
    last_euler    = zeros(3, 1);
    pos_0         = zeros(3, 1);
    euler_0       = zeros(3, 1);
end

% ---- Throttle to vo_hz -----------------------------------------
% Limit processing rate to configured VO frequency
update_interval = int32(round(1.0 / (vo_hz * Ts)));
counter = counter + int32(1);

if counter < update_interval
    pos_vo   = last_pos;
    euler_vo = last_euler;
    vo_valid = double(initialized);
    num_vis  = double(map_size);
    return;
end
counter = int32(0);

% ---- Build parameter structs -----------------------------------
% Pack camera, ORB, KLT, RANSAC and keyframe thresholds into structs
cam_p.fx          = cam_fx;
cam_p.fy          = cam_fy;
cam_p.cx          = cam_cx;
cam_p.cy          = cam_cy;
cam_p.width       = cam_width;
cam_p.height      = cam_height;
cam_p.blob_radius = blob_radius;
cam_p.blob_sigma  = blob_sigma;
cam_p.min_depth   = min_depth;
cam_p.max_depth   = max_depth;

orb_p.num_points   = orb_num_points;
orb_p.num_levels   = orb_num_levels;
orb_p.scale_factor = orb_scale_factor;

klt_p.block_size     = [klt_block_size_r, klt_block_size_c];
klt_p.max_iterations = klt_max_iter;
klt_p.num_levels     = klt_num_levels;
klt_p.max_error      = klt_max_error;

ran_p.max_distance   = ransac_max_dist;
ran_p.confidence     = ransac_conf;
ran_p.max_iterations = ransac_max_iter;

kf_p.min_features    = kf_min_feat;
kf_p.max_translation = kf_max_trans;
kf_p.max_rotation    = kf_max_rot;

landmarks = landmarks_3D_in(1:num_landmarks, :);

% ================================================================
% PHASE A: Waiting for map initialisation
% ================================================================
if ~initialized

    frame_curr = generate_camera_frame(...
        pos_true, euler_true, landmarks', cam_p);

    if ~init_checking
        frame_0    = frame_curr;
        pos_0      = pos_true;
        euler_0    = euler_true;
        kp_temp    = detect_orb_features(frame_0, orb_p);
        n0         = size(kp_temp, 1);
        kp_0       = zeros(500, 2);
        kp_0(1:n0,:) = kp_temp;
        frame_prev    = frame_0;
        frame_kf      = frame_0;
        init_checking = true;

        pos_vo=last_pos; euler_vo=last_euler;
        vo_valid=0; num_vis=0;
        return;
    end

    [ready, dist] = check_baseline(pos_0, pos_true, min_baseline_m);
    if ~ready
        pos_vo=last_pos; euler_vo=last_euler;
        vo_valid=0; num_vis=0;
        return;
    end

    fprintf('[INIT] Baseline ready: %.3fm\n', dist);

    kp_0_active = kp_0(any(kp_0~=0,2), :);
    [kp2, val, ~] = track_klt_features(frame_0, frame_curr, kp_0_active, klt_p);

    if sum(val) >= 8
        [p0, pK, ~, n_in] = ransac_filter(...
            kp_0_active(val,:), kp2(val,:), ran_p);

        if n_in >= 8
            [lm_cam0, ~, ~, ~, ~, ~, ok] = initialize_map(...
    frame_0, frame_curr, p0, pK, pos_0, pos_true, cam_p);
            if ok && size(lm_cam0,1) >= 8
                % Transform lm_cam0 → world NED
                R_be_0 = build_Rec(euler_0, R_cam_body)' * R_cam_body;
                % Simpler: build R_be_0 directly
                roll=euler_0(1); pitch=euler_0(2); yaw=euler_0(3);
                cr=cos(roll); sr=sin(roll); cp=cos(pitch);
                sp=sin(pitch); cy_=cos(yaw); sy_=sin(yaw);
                R_be_0=[cy_*cp, cy_*sp*sr-sy_*cr, cy_*sp*cr+sy_*sr;
                        sy_*cp, sy_*sp*sr+cy_*cr,  sy_*sp*cr-cy_*sr;
                        -sp,    cp*sr,               cp*cr           ];

                n_lm = size(lm_cam0,1);
                for i = 1:n_lm
                    p_body      = R_cam_body' * lm_cam0(i,:)';
                    p_world     = R_be_0 * p_body + pos_0;
                    map_3d(i,:) = p_world';
                end
                map_size = int32(n_lm);

                % Project map into current frame → kp_kf
                kp_kf = project_map_into_frame(...
                    map_3d, n_lm, pos_true, euler_true, ...
                    R_cam_body, cam_fx, cam_fy, cam_cx, cam_cy, ...
                    cam_width, cam_height, min_depth, max_depth);

                pos_kf      = pos_true;
                euler_kf    = euler_true;
                frame_kf    = frame_curr;
                frame_prev  = frame_curr;
                last_pos    = pos_true;
                last_euler  = euler_true;
                initialized = true;

                fprintf('[INIT] Map ready: %i landmarks\n', int32(n_lm));
            end
        end
    end

    pos_vo   = last_pos;
    euler_vo = last_euler;
    vo_valid = 0;
    num_vis  = double(map_size);
    return;
end

% ================================================================
% PHASE B: Map ready — run back-end every VO frame
% ================================================================
frame_curr   = generate_camera_frame(pos_true, euler_true, landmarks', cam_p);
cur_map_size = double(map_size);

% Check active projections WITHOUT slicing kp_kf — keep it fixed-size
n_active = sum(any(kp_kf(1:cur_map_size,:)~=0, 2));
if n_active < 6
    fprintf('[BACKEND] Too few active projections: %i\n', int32(n_active));
    pos_vo=last_pos; euler_vo=last_euler;
    vo_valid=1; num_vis=double(map_size);
    frame_prev = frame_curr;
    return;
end

% Run the back-end optimizer / tracking and update the map
[pos_out, euler_out, valid_out, map_3d, map_size, kp_kf] = run_backend(...
    frame_kf, frame_curr, ...
    kp_kf, pos_kf, euler_kf, ...
    pos_true, euler_true, ...
    map_3d, map_size, ...
    cam_p, R_cam_body, ...
    orb_p, klt_p, ran_p, kf_p, ...
    last_pos, last_euler);

if valid_out
    last_pos   = pos_out;
    last_euler = euler_out;

    d_pos   = norm(pos_true  - pos_kf);
    d_euler = norm(euler_true - euler_kf) * 180/pi;

    if d_pos > kf_p.max_translation || d_euler > kf_p.max_rotation

        fprintf('[KF] New keyframe — d_pos=%.2fm  d_euler=%.1fdeg\n', ...
            d_pos, d_euler);

        % Re-project map into current frame
        kp_kf = project_map_into_frame(...
            map_3d, double(map_size), pos_true, euler_true, ...
            R_cam_body, cam_fx, cam_fy, cam_cx, cam_cy, ...
            cam_width, cam_height, min_depth, max_depth);

        pos_kf   = pos_true;
        euler_kf = euler_true;
        frame_kf = frame_curr;
    end
end

frame_prev = frame_curr;
pos_vo     = last_pos;
euler_vo   = last_euler;
vo_valid   = double(valid_out);
num_vis    = double(map_size);
end