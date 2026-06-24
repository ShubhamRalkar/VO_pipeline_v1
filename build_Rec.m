function R_ec = build_Rec(euler, R_cam_body)
% BUILD_REC  Build earth-to-camera rotation matrix from euler angles
roll=euler(1); pitch=euler(2); yaw=euler(3);
cr=cos(roll); sr=sin(roll);
cp=cos(pitch); sp=sin(pitch);
cy_=cos(yaw);  sy_=sin(yaw);
R_be = [cy_*cp,  cy_*sp*sr-sy_*cr,  cy_*sp*cr+sy_*sr;
    sy_*cp,  sy_*sp*sr+cy_*cr,  sy_*sp*cr-cy_*sr;
    -sp,     cp*sr,              cp*cr            ];
R_ec = R_cam_body * R_be';
end