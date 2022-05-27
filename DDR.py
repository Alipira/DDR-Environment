import numpy as np 
from math import pi, cos, radians, sin


class DifferentialDriveRobot:
    def __init__(self, robot_pose:np.array, vrmax=0.26, wmax=1.82):
        """_summary_

        Args:
            robot_pose (np.array): _description_
            vrmax (float, optional): _description_. Defaults to 0.26.
            wmax (float, optional): _description_. Defaults to 1.82.
        """
        self.robot_pose = robot_pose
        self.vrmax = vrmax
        self.wmax = wmax

    def kinematic(self, vr:float, w_com:float, dt:float):
        """_summary_

        Args:
            vr (float): _description_
            w_com (float): _description_
            dt (float): _description_

        Returns:
            _type_: _description_
        """
        
        #dq
        vr = np.clip(vr, -self.vrmax, self.vrmax)
        w_com = np.clip(w_com, -self.wmax, self.wmax)

        vx = vr * cos(self.robot_pose[0, 2])  
        vy = vr * sin(self.robot_pose[0, 2])

        #Motion eqution based on Euler integration (q = q + dq * dt)
        self.robot_pose[0, 0] = self.robot_pose[0, 0] + vx * dt
        self.robot_pose[0, 1] = self.robot_pose[0, 1] + vy * dt
        self.robot_pose[0, 2] = self.robot_pose[0, 2] + w_com * dt 
        self.robot_pose[0, 2] = self.normalize_angle(self.robot_pose[0, 2])
        return np.array(self.robot_pose), vx, vy

    def normalize_angle(self, angle:radians):
        """This function takes robot's orientation and normalized it

        Args:
            angle (radians): robot's orientation

        Returns:
            radians: Normalized angle between [-pi, pi]
        """
        norm_angle = angle % (2 * pi)
        if norm_angle > pi:
            norm_angle -= 2 * pi
        return norm_angle

    def rest_pose(self,):
        """_summary_

        Args:
            theta (radians): _description_

        Returns:
            _type_: _description_
        """
        self.robot_pose[0, 0] = 0
        self.robot_pose[0, 1] = 0
        self.robot_pose[0, 2] = 0
        return np.array(self.robot_pose)
