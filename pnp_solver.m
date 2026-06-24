function [pos_est, euler_est, valid] = pnp_solver(pts_2d, pts_3d, ...
                                        cam_params, R_cam_body, ...
                                        last_pos, last_euler, euler_hint)
% euler_hint: use true euler for bearing transform (pass euler_curr from backend)
%             if not provided, falls back to last_euler

    if nargin < 7
        euler_hint = last_euler;
    end
% PNP_SOLVER  Ray Intersection + Wahba pose estimation
%             Extracted from vo_sensor.m — same math, new input source
%
% Inputs:
%   pts_2d      (Nx2) pixel coordinates [u, v] — from KLT tracker
%   pts_3d      (Nx3) world NED positions — from map
%   cam_params  struct: .fx .fy .cx .cy
%   R_cam_body  (3×3) camera-to-body rotation
%   last_pos    (3×1) previous position estimate (fallback)
%   last_euler  (3×1) previous euler estimate   (fallback)
%
% Outputs:
%   pos_est    (3×1) estimated NED position
%   euler_est  (3×1) estimated euler [roll; pitch; yaw]
%   valid      logical

    pos_est   = last_pos;
    euler_est = last_euler;
    valid     = false;

    N = size(pts_2d, 1);
    if N < 6
        return;
    end

    %-- Step 1: Compute bearing vectors in camera frame -------------
    bearings_cam = zeros(N, 3);
    for i = 1:N
        xn = (pts_2d(i,1) - cam_params.cx) / cam_params.fx;
        yn = (pts_2d(i,2) - cam_params.cy) / cam_params.fy;
        b  = [xn; yn; 1.0];
        bearings_cam(i,:) = (b / norm(b))';
    end

%-- Step 2: Build R_ec from euler_hint -------------------------
roll=euler_hint(1); pitch=euler_hint(2); yaw=euler_hint(3);
cr=cos(roll); sr=sin(roll);
cp=cos(pitch); sp=sin(pitch);
cy_=cos(yaw);  sy_=sin(yaw);

R_be = [cy_*cp,  cy_*sp*sr-sy_*cr,  cy_*sp*cr+sy_*sr;
        sy_*cp,  sy_*sp*sr+cy_*cr,  sy_*sp*cr-cy_*sr;
        -sp,     cp*sr,              cp*cr            ];
R_ec = R_cam_body * R_be';

    %-- Step 3: Ray Intersection — position estimation --------------
    % Identical to vo_sensor.m Steps 5
    M_ray = zeros(3,3);
    v_ray = zeros(3,1);

    for i = 1:N
        b_cam_i = bearings_cam(i,:)';
        b_world = R_ec' * b_cam_i;
        b_world = b_world / max(norm(b_world), 1e-6);

        L      = pts_3d(i,:)';
        bbT    = b_world * b_world';
        ImbbT  = eye(3) - bbT;

        M_ray = M_ray + ImbbT;
        v_ray = v_ray + ImbbT * L;
    end

    if abs(det(M_ray)) < 1e-4
        return;   % degenerate — return fallback
    end

    pos_est = M_ray \ v_ray;

    % Sanity check — reject wild jumps
    if norm(pos_est - last_pos) > 5.0
        pos_est = last_pos;
        return;
    end

    %-- Step 4: Wahba's Problem — rotation estimation ---------------
    % Identical to vo_sensor.m Step 6
    B_wahba = zeros(3,3);
    valid_dirs = 0;

    for i = 1:N
        d_world = pts_3d(i,:)' - pos_est;
        d_len   = norm(d_world);
        if d_len > 0.5
            d_world = d_world / d_len;
            b_cam_i = bearings_cam(i,:)';
            B_wahba = B_wahba + b_cam_i * d_world';
            valid_dirs = valid_dirs + 1;
        end
    end

    if valid_dirs < 4
        euler_est = last_euler;
        valid     = true;   % position OK, rotation fallback
        return;
    end

    [U_w, ~, V_w] = svd(B_wahba);
    U_w = real(U_w);
    V_w = real(V_w);

    D_corr = eye(3);
    if det(U_w) * det(V_w) < 0
        D_corr(3,3) = -1;
    end
    R_est = U_w * D_corr * V_w';

    %-- Step 5: R_est → euler angles --------------------------------
    % Identical to vo_sensor.m Step 7
    R_be_est = real(R_est' * R_cam_body);

    pitch_est = real(-asin(max(-1.0, min(1.0, R_be_est(3,1)))));
    cp_est    = cos(pitch_est);

    if abs(cp_est) > 1e-6
        roll_est = real(atan2(R_be_est(3,2)/cp_est, R_be_est(3,3)/cp_est));
        yaw_est  = real(atan2(R_be_est(2,1)/cp_est, R_be_est(1,1)/cp_est));
    else
        roll_est = 0.0;
        yaw_est  = real(atan2(-R_be_est(1,2), R_be_est(2,2)));
    end

    euler_est = [roll_est; pitch_est; yaw_est];
    valid     = true;
end