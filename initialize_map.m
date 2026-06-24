function [landmarks_3D, poses_R1, poses_t1, poses_R2, poses_t2, scale, success, num_lm_out] = initialize_map(...
           frame_0, frame_K, pts_0, pts_K, pos_0, pos_K, cam_params)
%#codegen
% INITIALIZE_MAP  Bootstrap 3D map from two keyframes — codegen compatible

    landmarks_3D = zeros(500, 3);
    poses_R1     = eye(3);
    poses_t1     = zeros(3,1);
    poses_R2     = eye(3);
    poses_t2     = zeros(3,1);
    scale        = 1.0;
    success      = false;
    num_lm_out   = int32(0);

    MIN_POINTS = int32(8);
    MAX_LM     = int32(500);

    n0 = int32(size(pts_0,1));
    nK = int32(size(pts_K,1));
    fprintf('[INIT] pts_0: %i,  pts_K: %i\n', n0, nK);

    if n0 < MIN_POINTS || nK < MIN_POINTS
        fprintf('[INIT] FAIL: not enough input points\n');
        return;
    end

    K = [cam_params.fx, 0,             cam_params.cx;
         0,             cam_params.fy,  cam_params.cy;
         0,             0,              1            ];

    intrinsics = cameraIntrinsics([cam_params.fx, cam_params.fy], ...
                                   [cam_params.cx, cam_params.cy], ...
                                   [cam_params.height, cam_params.width]);

    fprintf('[INIT] Computing Essential Matrix...\n');

    [E, ess_inliers] = estimateEssentialMatrix(pts_0, pts_K, intrinsics, ...
        'Confidence',   99.0, ...
        'MaxNumTrials', 1000, ...
        'MaxDistance',  1.0);

    num_ess_inliers = int32(sum(ess_inliers));
    fprintf('[INIT] Essential Matrix inliers: %i / %i\n', num_ess_inliers, n0);

    if num_ess_inliers < MIN_POINTS
        fprintf('[INIT] FAIL: too few Essential Matrix inliers\n');
        return;
    end

    pts0_in = pts_0(ess_inliers, :);
    ptsK_in = pts_K(ess_inliers, :);
    N_ess   = size(pts0_in, 1);

    fprintf('[INIT] Decomposing Essential Matrix...\n');

    [U, ~, V] = svd(E);
    E_clean   = U * diag([1,1,0]) * V';
    [U, ~, V] = svd(E_clean);

    W_mat = [0 -1 0; 1 0 0; 0 0 1];
    R1 = U *  W_mat  * V';
    R2 = U *  W_mat' * V';
    t_pos =  U(:,3);
    t_neg = -U(:,3);

    if det(R1) < 0, R1 = -R1; end
    if det(R2) < 0, R2 = -R2; end

    cand_R = zeros(3,3,4);
    cand_t = zeros(3,4);
    cand_R(:,:,1)=R1; cand_t(:,1)=t_pos;
    cand_R(:,:,2)=R1; cand_t(:,2)=t_neg;
    cand_R(:,:,3)=R2; cand_t(:,3)=t_pos;
    cand_R(:,:,4)=R2; cand_t(:,4)=t_neg;

    fprintf('[INIT] Testing 4 pose candidates...\n');

    best_count = int32(-1);
    best_R     = eye(3);
    best_t     = zeros(3,1);

    P0_base = K * [eye(3), zeros(3,1)];

    for c = 1:4
        R_c  = cand_R(:,:,c);
        t_c  = cand_t(:,c);
        PK_c = K * [R_c, t_c];

        count_front = int32(0);
        for i = 1:N_ess
            u0=pts0_in(i,1); v0=pts0_in(i,2);
            uK=ptsK_in(i,1); vK=ptsK_in(i,2);

            A = [u0*P0_base(3,:)-P0_base(1,:);
                 v0*P0_base(3,:)-P0_base(2,:);
                 uK*PK_c(3,:)-PK_c(1,:);
                 vK*PK_c(3,:)-PK_c(2,:)];

            [~,~,Vt] = svd(A);
            X_h = Vt(:,end);
            if abs(X_h(4)) < 1e-10
                continue;
            end
            X = X_h(1:3)/X_h(4);

            depth0 = X(3);
            XK_vec = R_c*X + t_c;
            depthK = XK_vec(3);

            if depth0 > 0 && depthK > 0
                count_front = count_front + int32(1);
            end
        end

        fprintf('[INIT]   Candidate %i: %i points in front\n', int32(c), count_front);

        if count_front > best_count
            best_count = count_front;
            best_R     = R_c;
            best_t     = t_c;
        end
    end

    fprintf('[INIT] Best candidate: %i / %i\n', best_count, int32(N_ess));

    if best_count < MIN_POINTS
        fprintf('[INIT] FAIL: no good pose candidate\n');
        return;
    end

    R_rel = best_R;
    t_rel = best_t;

    fprintf('[INIT] Triangulating...\n');

    P0 = K * [eye(3), zeros(3,1)];
    PK = K * [R_rel,  t_rel      ];

    N_in = size(pts0_in, 1);
    idx  = int32(0);

    for i = 1:N_in
        if idx >= MAX_LM
            break;
        end

        u0=pts0_in(i,1); v0=pts0_in(i,2);
        uK=ptsK_in(i,1); vK=ptsK_in(i,2);

        A = [u0*P0(3,:)-P0(1,:);
             v0*P0(3,:)-P0(2,:);
             uK*PK(3,:)-PK(1,:);
             vK*PK(3,:)-PK(2,:)];

        [~,~,V_svd] = svd(A);
        X_h = V_svd(:,end);
        if abs(X_h(4)) < 1e-10
            continue;
        end
        X = X_h(1:3)/X_h(4);

        XK_vec = R_rel*X + t_rel;
        depth0 = X(3);
        depthK = XK_vec(3);

        if depth0 > 0 && depthK > 0
            idx = idx + int32(1);
            landmarks_3D(idx,:) = X';
        end
    end

    num_valid = idx;
    fprintf('[INIT] Triangulated: %i / %i valid\n', num_valid, int32(N_in));

    if num_valid < MIN_POINTS
        fprintf('[INIT] FAIL: too few triangulated points\n');
        return;
    end

    imu_displacement = norm(pos_K - pos_0);
    vis_translation  = norm(t_rel);

    if imu_displacement < 0.05
        fprintf('[INIT] FAIL: IMU baseline too small\n');
        return;
    end

    scale = imu_displacement / vis_translation;
    fprintf('[INIT] Scale: %.4f\n', scale);

    for i = 1:double(num_valid)
        landmarks_3D(i,:) = landmarks_3D(i,:) * scale;
    end

    poses_R1 = eye(3);
    poses_t1 = zeros(3,1);
    poses_R2 = R_rel;
    poses_t2 = t_rel * scale;

    num_lm_out = num_valid;
    success    = true;
    fprintf('[INIT] SUCCESS — %i landmarks\n', num_lm_out);
end