function pts2d = project_map_into_frame(map_pts, n_pts, pos, euler, ...
    R_cam_body, fx, fy, cx, cy, ...
    W, H, dmin, dmax)
% PROJECT_MAP_INTO_FRAME  Project 3D map points into image plane
%
% Returns (2000×2) matrix — row i = [u,v] of map point i
% Zero rows = landmark not visible in this frame

pts2d = zeros(2000, 2);
R_ec  = build_Rec(euler, R_cam_body);

for i = 1:n_pts
    L_cam = R_ec * (map_pts(i,:)' - pos);
    if L_cam(3) < dmin || L_cam(3) > dmax, continue; end
    u = fx * L_cam(1)/L_cam(3) + cx;
    v = fy * L_cam(2)/L_cam(3) + cy;
    if u >= 1 && u <= W && v >= 1 && v <= H
        pts2d(i,:) = [u, v];
    end
end
end