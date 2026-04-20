"""implements Whole-Body Control logic with MPC"""

import casadi as ca
from urdf2casadi import urdfparser as u2c
import numpy as np
from scipy.spatial.transform import Rotation
import time
import os.path as osp
import matplotlib.pyplot as plt
import logging, sys
import argparse

np.set_printoptions(precision=3, suppress=True)


class WBC:
    def __init__(self, path_urdf, tf_root="locobot/base_footprint", tf_tip="locobot/ee_gripper_link"):
        # Load URDF
        parser = u2c.URDFparser()
        parser.from_file(path_urdf)

        # Build forward kinematics expressions
        joint_list, joints_name, q_max, q_min = parser.get_joint_info(tf_root, tf_tip)
        q_max = np.array(q_max)
        q_min = np.array(q_min)
        n_joints = parser.get_n_joints(tf_root, tf_tip)
        assert n_joints == 6, "Expected 6 joints for locobot arm, got {}".format(n_joints)
        print(f"joints name: {joints_name}\nq_max: {q_max}\nq_min: {q_min}")
        self.fk_dict = parser.get_forward_kinematics(tf_root, tf_tip)

        n = 3 + 6  # (x, y, theta, q1, q2, ..., q6)

        # define CasADi optimization problem
        self.opti = ca.Opti()
        # goal pose
        self.T_ref = self.opti.parameter(4, 4)
        self.x0 = self.opti.parameter(n)
        # variables to solve
        self.x = self.opti.variable(n)

        # constrain gripper pose
        T = self.get_gripper_pose(self.x)
        self.opti.subject_to(ca.sumsqr(T - self.T_ref) <= 1e-4)  # position and orientation error within threshold
        self.opti.subject_to(self.opti.bounded(q_min, self.x[3:], q_max))  # joint limits
        self.opti.subject_to(self.opti.bounded(-np.pi, self.x[2], np.pi))

        # set joints cost
        self.opti.minimize(ca.sumsqr((self.x - self.x0)[3:]))

        # set solver options
        opts_setting = {
            "jit": True,
            "print_time": 0,
            "ipopt.print_level": 3,  # Set to 0 for less verbose output
        }
        self.opti.solver("ipopt", opts_setting)

    def pose2d_to_mat(self, x, y, theta):
        """CasADi version of pose2d_to_mat"""
        R = ca.vertcat(
            ca.horzcat(ca.cos(theta), -ca.sin(theta), 0),
            ca.horzcat(ca.sin(theta), ca.cos(theta), 0),
            ca.horzcat(0, 0, 1),
        )
        t = ca.vertcat(x, y, -0.242)
        T = ca.vertcat(ca.horzcat(R, t), ca.horzcat(0, 0, 0, 1))
        return T

    def get_gripper_pose(self, X):
        """Get EE pose from original state (first 9 elements)"""
        T_b2w = self.pose2d_to_mat(X[0], X[1], X[2])
        T_ee2b = self.fk_dict["T_fk"](X[3:9])
        return ca.mtimes(T_b2w, T_ee2b)

    def solve(self, T_ref, x0=None):
        if x0 is None:
            x0 = np.zeros(9)
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.T_ref, T_ref)
        self.sol = self.opti.solve()
        x = self.sol.value(self.x)
        return x

if __name__ == "__main__":
    wbc = WBC()
    T_ref = np.eye(4)
    T_ref[:3, :3] = Rotation.from_euler("xyz", [np.pi/4, 0, 0]).as_matrix()
    T_ref[:3, 3] = [20, 10, 0]
    x = wbc.solve(T_ref)
    print(x)
    print("\nGripper pose from solution:", wbc.get_gripper_pose(x))