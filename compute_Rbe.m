function R_be = compute_Rbe(euler)
% COMPUTE_RBE  Build body-to-earth DCM from euler angles [roll;pitch;yaw]
roll=euler(1); pitch=euler(2); yaw=euler(3);
cr=cos(roll);  sr=sin(roll);
cp=cos(pitch); sp=sin(pitch);
cy=cos(yaw);   sy=sin(yaw);
R_be = [cp*cy,          cp*sy,         -sp;
    sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, sr*cp;
    cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cr*cp];
end