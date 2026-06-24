%% drone_params.m — Quadrotor Physical Parameters
% Run this script BEFORE opening the Simulink model

%--- Mass & Geometry ---
m     = 1.5;        % Total mass [kg]
g     = 9.81;       % Gravity [m/s²]
L     = 0.25;       % Arm length from center to rotor [m]

%--- Inertia Matrix (diagonal, symmetric quadrotor) ---
Ixx   = 0.0232;     % Moment of inertia about X-axis [kg·m²]
Iyy   = 0.0232;     % Moment of inertia about Y-axis [kg·m²]
Izz   = 0.0468;     % Moment of inertia about Z-axis [kg·m²]
I_body = diag([Ixx, Iyy, Izz]);  % 3x3 inertia matrix

%--- Rotor Parameters ---
kT    = 1.2e-5;     % Thrust coefficient [N/(rad/s)²]
kD    = 1.5e-7;     % Drag/torque coefficient [N·m/(rad/s)²]
w_max = 1200;       % Max rotor speed [rad/s]

%--- Hover Condition ---
w_hover = sqrt(m * g / (4 * kT));  % Rotor speed for hover [rad/s]
fprintf('Hover rotor speed: %.1f rad/s\n', w_hover);
fprintf('Hover rotor speed: %.0f RPM\n', w_hover * 60/(2*pi));

%--- Simulation ---
Ts = 0.001;         % Simulation timestep [s]
T_end = 20;         % Simulation duration [s]

%--- Initial Conditions ---
pos_0   = [0; 0; 0];       % Initial position [x, y, z] in NED [m]
vel_0   = [0; 0; 0];       % Initial velocity [m/s]
euler_0 = [0; 0; 0];       % Initial orientation [roll, pitch, yaw] [rad]
omega_0 = [0; 0; 0];       % Initial angular velocity [rad/s]

%------------Phase2-------------------onwards__________________>
%% Phase 2 PID Gains — add to bottom of drone_params.m

% Attitude controller gains (inner loop)
Kp_att = [6; 6; 4];     % roll, pitch, yaw
%Ki_att = [1; 1; 0.5];
Kd_att = [3.5; 3.5; 2];

% Position controller gains (outer loop)
Kp_pos = [0.5; 0.5; 1.0];
%Kd_pos = [0.8; 0.8; 1.5];   % increase Z damping especially
Kd_pos = [0.8; 1.2; 1.8];   % more damping on Y and Z
%Ki_pos = [0.05; 0.05; 0.1];
%Kd_pos = [0.8; 0.8; 0.9];

%% Phase 3 — Trajectory Generator parameters
traj_start = [0;  0;   0];    % start position [m]
traj_goal  = [10; 9; -10];    % goal position [m]
traj_speed = 1.0;             % travel speed [m/s]
%% Phase 4 — IMU Sensor Model Parameters

% Accelerometer noise and bias
accel_noise_std  = 0.007;   % std dev of white noise [m/s²]
accel_bias_init  = [0.05; -0.03; 0.04];  % initial constant bias [m/s²]
accel_drift_std  = 0.0001;  % std dev of bias random walk per step

% Gyroscope noise and bias  
gyro_noise_std   = 0.005;  % std dev of white noise [rad/s]
gyro_bias_init   = [0.002; -0.001; 0.003];  % initial bias [rad/s]
gyro_drift_std   = 0.0001; % std dev of bias random walk per step
%% Phase 5 — VIO Sensor Model Parameters

vio_noise_std    = 0.05;   % position noise std dev [m]
vio_update_hz    = 30;     % VIO camera update rate [Hz]
vio_dropout_prob = 0.02;   % probability of dropout per update (2%)
%% Phase 6 EKF — upgraded to 15 states
Q_ekf = diag([1e-4, 1e-4, 1e-4, ...     % position
              1e-3, 1e-3, 1e-3, ...     % velocity
              1e-4, 1e-4, 1e-4, ...     % attitude
              1e-9, 1e-9, 1e-9, ...     % accel bias  — WAS 1e-4, NOW 1e-9
              1e-9, 1e-9, 1e-9]);       % gyro bias   — WAS 1e-5, NOW 1e-9

P0 = diag([0.1, 0.1, 0.1, ...           % position
           0.1, 0.1, 0.1, ...           % velocity
           0.01, 0.01, 0.01, ...        % attitude
           0.01, 0.01, 0.01, ...        % accel bias — WAS 1.0, NOW 0.01
           0.01, 0.01, 0.01]);          % gyro bias  — WAS 0.1, NOW 0.01

% Trust VIO more → stronger corrections

%% ============================================
%% Phase 9: Visual Odometry Parameters
%% ============================================


%% Camera Intrinsics (Monocular, forward-facing)
cam_fx = 320;           % focal length x [pixels]
cam_fy = 320;           % focal length y [pixels]
cam_cx = 320;           % principal point x [pixels]
cam_cy = 240;           % principal point y [pixels]
cam_width  = 640;       % image width [pixels]
cam_height = 480;       % image height [pixels]
pixel_noise_std = 1.0;  % pixel measurement noise [pixels]
vo_update_hz = 30;      % camera frame rate [Hz]
min_features = 6;       % minimum features for pose estimation

% Camera-to-body rotation (camera Z = body X = forward)
R_cam_body = [0 1 0; 0 0 1; 1 0 0];

%% EKF (updated for VO — 6 measurements: position + euler)
R_ekf = diag([0.08^2, 0.08^2, 0.08^2, ...    % VO position noise
    0.05^2, 0.05^2, 0.05^2]);       % VO euler noise
% P0, Q_ekf, x0_ekf remain the same 15-state versions

%% ============================================
%% Phase 10: City Environment + Circular Trajectory
%% ============================================

%% Circular Trajectory Parameters
circle_center_x = 0;
circle_center_y = 0;
circle_radius   = 15;          % [m]
circle_alt      = -10;         % [m] NED (10m above ground)
circle_speed    = 1.0;         % [m/s]
takeoff_time    = 12;          % [s] time to climb to altitude
T_end           = 110;         % [s] total simulation time

%% City Buildings Definition
% Each row: [center_x, center_y, width_x, depth_y, height]
% Buildings arranged around the circular path (radius 25-35m from center)
buildings_def = [
    28,   0,    8,  6, 12;     % East
    20,  20,   10,  8, 10;     % North-East
     0,  28,    6, 10, 15;     % North
   -20,  20,    8,  8, 11;     % North-West
   -28,   0,   10,  6, 13;     % West
   -20, -20,    8, 10,  9;     % South-West
     0, -28,   10,  8, 14;     % South
    20, -20,    6,  8, 10;     % South-East
    35,  15,    6,  6,  8;     % Extra NE
   -35, -15,    8,  6, 10;     % Extra SW
];

%% Generate City Landmarks
city_landmarks = [];

% ---- 1. Building features (walls = windows) ----
for b = 1:size(buildings_def, 1)
    bx = buildings_def(b, 1);
    by = buildings_def(b, 2);
    bw = buildings_def(b, 3);    % width (X)
    bd = buildings_def(b, 4);    % depth (Y)
    bh = buildings_def(b, 5);    % height

    % Front wall (X + w/2): windows every 2m horizontal, 3m vertical
    for wy = (-bd/2):2:(bd/2)
        for wz = -3:-3:(-bh)
            city_landmarks = [city_landmarks; bx + bw/2, by + wy, wz];
        end
    end

    % Back wall (X - w/2)
    for wy = (-bd/2):2:(bd/2)
        for wz = -3:-3:(-bh)
            city_landmarks = [city_landmarks; bx - bw/2, by + wy, wz];
        end
    end

    % Left wall (Y + d/2)
    for wx = (-bw/2):2:(bw/2)
        for wz = -3:-3:(-bh)
            city_landmarks = [city_landmarks; bx + wx, by + bd/2, wz];
        end
    end

    % Right wall (Y - d/2)
    for wx = (-bw/2):2:(bw/2)
        for wz = -3:-3:(-bh)
            city_landmarks = [city_landmarks; bx + wx, by - bd/2, wz];
        end
    end

    % Rooftop features (edges, corners, center)
    for rx = [-bw/2, 0, bw/2]
        for ry = [-bd/2, 0, bd/2]
            city_landmarks = [city_landmarks; bx + rx, by + ry, -bh];
        end
    end
end

% ---- 2. Road markings (ground features at Z=0) ----
road_pts = [];
% Radial roads (4 directions)
for angle = [0, 90, 180, 270] * pi/180
    for r = 5:3:40
        road_pts = [road_pts; r*cos(angle), r*sin(angle), 0];
    end
end
% Ring road around the flight path
for angle = 0:10:350
    a_rad = angle * pi / 180;
    for r = [10, 20, 30]
        road_pts = [road_pts; r*cos(a_rad), r*sin(a_rad), 0];
    end
end

% ---- 3. Streetlights (poles with lights at top) ----
light_pts = [];
for angle = 0:30:330
    a_rad = angle * pi / 180;
    lx = 22 * cos(a_rad);
    ly = 22 * sin(a_rad);
    for lz = 0:-1.5:-6       % pole features at multiple heights
        light_pts = [light_pts; lx, ly, lz];
    end
end

% ---- 4. Trees (scattered in parks between buildings) ----
rng(99);
tree_pts = [];
% Park areas (between buildings)
park_centers = [10, 10; -10, 10; -10, -10; 10, -10; 0, 0];
for p = 1:size(park_centers, 1)
    px = park_centers(p, 1);
    py = park_centers(p, 2);
    for k = 1:4    % 4 trees per park
        tx = px + 4*(rand()-0.5);
        ty = py + 4*(rand()-0.5);
        for tz = 0:-1:-4    % trunk + canopy features
            tree_pts = [tree_pts; tx, ty, tz];
        end
    end
end

% ---- 5. Ground texture (sidewalk corners, manholes, curbs) ----
ground_pts = [];
for gx = -35:5:35
    for gy = -35:5:35
        % Skip where buildings are (approximate)
        dist_from_center = sqrt(gx^2 + gy^2);
        if dist_from_center > 5 && dist_from_center < 40
            ground_pts = [ground_pts; gx, gy, 0];
        end
    end
end

% ---- Combine all landmarks ----
landmarks_3D = [city_landmarks; road_pts; light_pts; tree_pts; ground_pts];
num_landmarks = size(landmarks_3D, 1);

fprintf('\n=== City Environment ===\n');
fprintf('  Buildings:     %d features\n', size(city_landmarks, 1));
fprintf('  Roads:         %d features\n', size(road_pts, 1));
fprintf('  Streetlights:  %d features\n', size(light_pts, 1));
fprintf('  Trees:         %d features\n', size(tree_pts, 1));
fprintf('  Ground:        %d features\n', size(ground_pts, 1));
fprintf('  TOTAL:         %d landmarks\n', num_landmarks);
fprintf('========================\n\n');
%% ============================================================
%  MODULE 2 — MAP INITIALIZER PARAMETERS
%  ============================================================
min_baseline       = 0.5;    % m  — minimum displacement before init
min_init_points    = 8;      % minimum triangulated landmarks
max_reproj_error   = 2.0;    % px — discard landmarks above this
min_parallax_deg   = 1.0;    % degrees — minimum parallax for good triangulation