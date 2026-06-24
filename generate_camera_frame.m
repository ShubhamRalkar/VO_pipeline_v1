function frame = generate_camera_frame(pos_ned, euler, landmarks, cam_params)
% GENERATE_CAMERA_FRAME  Render synthetic grayscale frame from drone pose
%
% Inputs:
%   pos_ned   (3×1) drone position in NED frame [X;Y;Z]
%   euler     (3×1) [roll;pitch;yaw] in radians
%   landmarks (3×N) landmark positions in world NED frame
%   cam_params struct with fields:
%             .fx .fy .cx .cy         intrinsics
%             .width .height          image size
%             .blob_radius            rendered dot size (px)
%             .blob_sigma             Gaussian blur sigma
%             .min_depth .max_depth   visibility range
%
% Output:
%   frame     (height×width uint8) synthetic grayscale image

W = cam_params.width;
H = cam_params.height;

% Initialise blank dark-grey frame (not pure black — gives
% ORB a background gradient to work against)
frame_f = 30 * ones(H, W, 'single');

%-- 1. Build rotation: world NED → camera frame ------------------
roll  = euler(1);
pitch = euler(2);
yaw   = euler(3);

% DCM body→earth (R_be), earth→body = R_be'
cr = cos(roll);  sr = sin(roll);
cp = cos(pitch); sp = sin(pitch);
cy = cos(yaw);   sy = sin(yaw);

R_be = [cp*cy,            cp*sy,           -sp;
    sr*sp*cy-cr*sy,   sr*sp*sy+cr*cy,   sr*cp;
    cr*sp*cy+sr*sy,   cr*sp*sy-sr*cy,   cr*cp];

% Camera→body alignment (camera Z = body X = forward)
R_cam_body = [0 1 0; 0 0 1; 1 0 0];

% Full rotation: world NED → camera frame
R_ec = R_cam_body * R_be';   % earth→camera

%-- 2. Project each landmark onto image plane --------------------
N = size(landmarks, 2);

for i = 1:N
    % Transform landmark to camera frame
    L_world  = landmarks(:, i);
    L_cam    = R_ec * (L_world - pos_ned);

    depth = L_cam(3);   % Z in camera frame = depth

    % Visibility checks
    if depth < cam_params.min_depth || depth > cam_params.max_depth
        continue
    end

    % Pinhole projection
    u = cam_params.fx * L_cam(1) / depth + cam_params.cx;
    v = cam_params.fy * L_cam(2) / depth + cam_params.cy;

    % Image bounds check (with margin for blob radius)
    r = cam_params.blob_radius;
    if u < r+1 || u > W-r || v < r+1 || v > H-r
        continue
    end

    % Render landmark as Gaussian blob
    % Brightness inversely proportional to depth (closer = brighter)
    brightness = single(220 * exp(-depth / 20.0) + 35);

    ui = round(u);
    vi = round(v);

    % Draw blob in a local patch
    for dy = -r:r
        for dx = -r:r
            dist2 = dx^2 + dy^2;
            if dist2 <= r^2
                px = ui + dx;
                py = vi + dy;
                if px >= 1 && px <= W && py >= 1 && py <= H
                    gauss = exp(-dist2 / (2 * cam_params.blob_sigma^2));
                    frame_f(py, px) = min(255, ...
                        frame_f(py, px) + brightness * gauss);
                end
            end
        end
    end
end

% Add subtle Gaussian noise for texture (helps ORB)
noise = 4 * randn(H, W, 'single');
frame_f = frame_f + noise;
frame_f = max(0, min(255, frame_f));

frame = uint8(frame_f);
end