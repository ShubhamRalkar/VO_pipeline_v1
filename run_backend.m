function [pos_vo, euler_vo, vo_valid, map_3d, map_size, pts_2d_kf] = run_backend(...
            frame_kf, frame_curr, ...
            pts_2d_kf, pos_kf, euler_kf, ...
            pos_curr, euler_curr, ...
            map_3d, map_size, ...
            cam_params, R_cam_body, ...
            orb_params, klt_params, ransac_p, kf_params, ...
            last_pos, last_euler)
        %#codegen

        % Force fixed sizes explicitly — prevents codegen variable-size propagation
        assert(size(pts_2d_kf,1) == 2000);
        assert(size(pts_2d_kf,2) == 2);
        assert(size(map_3d,1) == 2000);
        assert(size(map_3d,2) == 3);
%#codegen
% RUN_BACKEND  Module 3 — codegen-compatible version

    MAX_MAP_SIZE = int32(2000);

    pos_vo   = last_pos;
    euler_vo = last_euler;
    vo_valid = false;

    if map_size < int32(6)
        return;
    end

    %-- Step 1: Filter to valid map projections ---------------------
    map_sz_d   = double(map_size);
    active_map = map_3d(1:map_sz_d, :);
    active_2d  = pts_2d_kf(1:map_sz_d, :);

    valid_proj      = active_2d(:,1) > 1 & active_2d(:,2) > 1;
    active_2d_filt  = active_2d(valid_proj, :);
    active_map_filt = active_map(valid_proj, :);
    tracking_size   = int32(size(active_2d_filt, 1));

    fprintf('[BACKEND] Valid map projections: %i / %i\n', tracking_size, map_size);

    if tracking_size < int32(6)
        fprintf('[BACKEND] Too few valid projections\n');
        return;
    end

    %-- Step 2: KLT track map projections keyframe  current --------
    [pts_curr_tracked, validity, num_tracked_d] = track_klt_features(...
        frame_kf, frame_curr, active_2d_filt, klt_params);
    num_tracked = int32(num_tracked_d);

    fprintf('[BACKEND] KLT tracked: %i / %i\n', num_tracked, tracking_size);

    if num_tracked < int32(6)
        fprintf('[BACKEND] Too few tracked points\n');
        return;
    end

    pts_2d_valid = pts_curr_tracked(validity, :);
    pts_3d_valid = active_map_filt(validity, :);

    %-- Step 3: PnP pose estimation ---------------------------------
    [pos_out, euler_out, ok] = pnp_solver(...
        pts_2d_valid, pts_3d_valid, cam_params, R_cam_body, ...
        last_pos, last_euler, euler_curr);

    fprintf('[BACKEND] PnP: valid=%i  pos=[%.2f, %.2f, %.2f]\n', ...
        int32(ok), pos_out(1), pos_out(2), pos_out(3));

    if ~ok
        fprintf('[BACKEND] PnP failed\n');
        return;
    end

    pos_vo   = pos_out;
    euler_vo = euler_out;
    vo_valid = true;

    %-- Step 4: Keyframe decision -----------------------------------
    d_pos   = norm(pos_curr  - pos_kf);
    d_euler = norm(euler_curr - euler_kf) * 180/pi;
    is_kf   = d_pos   > kf_params.max_translation || ...
              d_euler > kf_params.max_rotation    || ...
              num_tracked < int32(kf_params.min_features);

    %-- Step 5: Triangulate new landmarks on keyframe ---------------
    if is_kf && map_size < MAX_MAP_SIZE

        fprintf('[BACKEND] Keyframe: triangulating new landmarks\n');

        K = [cam_params.fx, 0,             cam_params.cx;
             0,             cam_params.fy,  cam_params.cy;
             0,             0,              1            ];

        roll=euler_kf(1); pitch=euler_kf(2); yaw=euler_kf(3);
        cr=cos(roll); sr=sin(roll); cp=cos(pitch);
        sp=sin(pitch); cy_=cos(yaw); sy_=sin(yaw);
        R_be_kf = [cy_*cp,  cy_*sp*sr-sy_*cr,  cy_*sp*cr+sy_*sr;
                   sy_*cp,  sy_*sp*sr+cy_*cr,   sy_*sp*cr-cy_*sr;
                   -sp,     cp*sr,               cp*cr            ];
        R_ec_kf = R_cam_body * R_be_kf';
        t_kf    = -R_ec_kf * pos_kf;
        P_kf    = K * [R_ec_kf, t_kf];

        roll=euler_curr(1); pitch=euler_curr(2); yaw=euler_curr(3);
        cr=cos(roll); sr=sin(roll); cp=cos(pitch);
        sp=sin(pitch); cy_=cos(yaw); sy_=sin(yaw);
        R_be_c = [cy_*cp,  cy_*sp*sr-sy_*cr,  cy_*sp*cr+sy_*sr;
                  sy_*cp,  sy_*sp*sr+cy_*cr,   sy_*sp*cr-cy_*sr;
                  -sp,     cp*sr,               cp*cr            ];
        R_ec_c = R_cam_body * R_be_c';
        t_cur  = -R_ec_c * pos_curr;
        P_cur  = K * [R_ec_c, t_cur];

        new_kp  = detect_orb_features(frame_curr, orb_params);
        n_added = int32(0);
        n_new_pts = size(new_kp, 1);

        tracked_valid    = pts_curr_tracked(validity,:);
        n_tracked_valid  = size(tracked_valid, 1);

        for i = 1:n_new_pts
            if map_size >= MAX_MAP_SIZE
                break;
            end

            uK = new_kp(i,1);
            vK = new_kp(i,2);

            % Skip if too close to an already-tracked point
            too_close = false;
            if n_tracked_valid > 0
                dists = sqrt((tracked_valid(:,1)-uK).^2 + (tracked_valid(:,2)-vK).^2);
                if min(dists) < 5.0
                    too_close = true;
                end
            end
            if too_close
                continue;
            end

            u0 = uK;
            v0 = vK;

            A = [u0*P_kf(3,:) - P_kf(1,:);
                 v0*P_kf(3,:) - P_kf(2,:);
                 uK*P_cur(3,:) - P_cur(1,:);
                 vK*P_cur(3,:) - P_cur(2,:)];

            [~,~,V_t] = svd(A);
            X_h = V_t(:,end);
            if abs(X_h(4)) < 1e-10
                continue;
            end
            X_w = X_h(1:3) / X_h(4);

            Xc_kf_vec  = R_ec_kf * X_w + t_kf;
            Xc_cur_vec = R_ec_c  * X_w + t_cur;
            if Xc_kf_vec(3)  < 0.5 || Xc_kf_vec(3)  > 80
                continue;
            end
            if Xc_cur_vec(3) < 0.5 || Xc_cur_vec(3) > 80
                continue;
            end

            map_size = map_size + int32(1);
            map_3d(map_size,:)    = X_w';
            pts_2d_kf(map_size,:) = [u0, v0];
            n_added = n_added + int32(1);
        end

        if n_added > int32(0)
            fprintf('[BACKEND] Added %i landmarks — map=%i\n', n_added, map_size);
        end
    end

    %-- Step 6: Map size cap (FIFO) — fixed size, no resize ---------
    if map_size > MAX_MAP_SIZE
        keep_d   = double(map_size - MAX_MAP_SIZE + int32(1));
        sz_d     = double(map_size);
        map_3d(1:double(MAX_MAP_SIZE),:)    = map_3d(keep_d:sz_d,:);
        pts_2d_kf(1:double(MAX_MAP_SIZE),:) = pts_2d_kf(keep_d:sz_d,:);
        map_size = MAX_MAP_SIZE;
    end
end