quaternions = csvread('matlab_quaternions.csv');

euler = quat2eul(quaternions);

euler = [euler(:,3), euler(:,2), euler(:,1)]; % swap order to be [roll (x), pitch (y), yaw (z)]

csvwrite('matlab_euler.csv', [quaternions, rad2deg(euler)]);
