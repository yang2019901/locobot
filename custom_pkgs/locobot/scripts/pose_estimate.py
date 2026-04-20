#!/usr/bin/env python3

## Provides a ros node class for pose estimation
## assuming you've got snapshots of the target (using scan.py)
## it reads in RGBD frames and publishs target's pose

import numpy as np
import open3d as o3d
import os
import cv2
import threading
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
import argparse
import json

import rospy
import tf, tf2_ros
import tf.transformations
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion

from GMatch import gmatch



class PoseEstimator:
    def __init__(self, args):
        self.args = args

        self.imgs_src, self.clds_src, self.masks_src, self.poses_src = (
            None,
            None,
            None,
            None,
        )
        self.lock_snapshots = threading.Lock()

        self.lock_rgb = threading.Lock()
        self.lock_depth = threading.Lock()
        self.img_rgb = None
        self.img_dep = None

        cam_info: CameraInfo = rospy.wait_for_message(
            "/locobot/camera/color/camera_info", CameraInfo
        )
        self.cam_intrin = np.array(cam_info.K).reshape((3, 3))

        self.coord_targ = "pose_estimate/target"
        self.coord_cam = "locobot/camera_color_optical_frame"  ## not locobot/camera_aligned_depth_to_color_frame

        self.bridge = CvBridge()
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)
        self.tf_pub = tf2_ros.TransformBroadcaster()

        rospy.Subscriber("/locobot/camera/color/image_raw", Image, self.on_rec_img)
        rospy.Subscriber(
            "/locobot/camera/aligned_depth_to_color/image_raw", Image, self.on_rec_depth
        )

        while self.img_rgb is None or self.img_dep is None:
            rospy.sleep(0.01)

    def reset_obj(self, path_obj):
        with self.lock_snapshots:
            with open(path_obj, "rb") as f:
                snapshots = pickle.load(f)
            self.imgs_src, self.clds_src, self.masks_src, M_ex_list = zip(*snapshots)
            self.poses_src = [gmatch.util.mat2pose(M_ex) for M_ex in M_ex_list]
            self.cache_id = path_obj

    def on_rec_img(self, msg):
        with self.lock_rgb:
            self.img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def on_rec_depth(self, msg):
        with self.lock_depth:
            self.img_dep = (
                self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1") * 1e-3
            )

    def run_once(self):
        """return err_code, err_msg"""
        with self.lock_rgb:
            img_rgb = deepcopy(self.img_rgb)
        with self.lock_depth:
            img_dep = deepcopy(self.img_dep)
        if img_rgb is None or img_dep is None:
            return -1, "img_rgb or img_dep is not ready."
        if self.imgs_src is None:
            return -1, "Source images are None, maybe not loaded."

        self.M_m2c = None

        D_near = 1e-2
        D_far = 1

        with self.lock_snapshots:
            match_data = gmatch.util.MatchData(
                imgs_src=self.imgs_src,
                clds_src=self.clds_src,
                masks_src=self.masks_src,
                poses_src=self.poses_src,
                img_dst=img_rgb,
                cld_dst=gmatch.util.depth2cld(img_dep, self.cam_intrin),
                mask_dst=np.where(
                    (img_dep > D_near) & (img_dep < D_far), 255, 0
                ).astype(np.uint8),
            )

            t0 = rospy.Time.now()
            gmatch.Match(match_data, cache_id=self.cache_id, debug=self.args.debug)
            gmatch.util.Solve(match_data)
            # gmatch.util.Refine(match_data)

        L = len(match_data.matches_list[match_data.idx_best])
        rospy.loginfo(f"gmatch costs {(rospy.Time.now() - t0)*1e-6} ms. match len: {L}")
        if L < self.args.minL:
            return (
                -1,
                f"Not enough matches. current size ({L}) < minL ({self.args.minL})",
            )

        self.M_m2c = match_data.mat_m2c
        pos, rot = gmatch.util.mat2pose(self.M_m2c)

        msg = TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.coord_cam
        msg.child_frame_id = self.coord_targ
        msg.transform.translation = Vector3(*pos)
        msg.transform.rotation = Quaternion(*rot)

        self.tf_pub.sendTransform(msg)
        return 0, "success"

    def run(self, M_ee2m=None):
        r = rospy.Rate(3)
        while not rospy.is_shutdown():
            self.run_once()

            if M_ee2m is not None and self.M_m2c is not None:
                M_ee2c = self.M_m2c @ M_ee2m
                R, t = M_ee2c[:3, :3], M_ee2c[:3, 3]
                K = self.cam_intrin
                axes = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]).T
                axes_cam = R @ axes + t.reshape(3, 1)
                axes_pix = (K @ axes_cam).T
                axes_pix = (axes_pix[:, :2].T / axes_pix[:, 2]).T.astype(np.int32)
                origin_pix = (K @ t).reshape(3)
                origin_pix = (origin_pix[:2] / origin_pix[2]).astype(np.int32)
                cv2.arrowedLine(
                    self.img_rgb, tuple(origin_pix), tuple(axes_pix[0]), (0, 0, 255), 3
                )  # X in red
                cv2.arrowedLine(
                    self.img_rgb, tuple(origin_pix), tuple(axes_pix[1]), (0, 255, 0), 3
                )  # Y in green
                cv2.arrowedLine(
                    self.img_rgb, tuple(origin_pix), tuple(axes_pix[2]), (255, 0, 0), 3
                )  # Z in blue
            cv2.imshow("rgb", cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imshow("depth", self.img_dep)
            key = cv2.waitKey(1)
            if key == ord("s"):
                print("save color.png and depth.png")
                cv2.imwrite(
                    "cache/color.png", cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    "cache/depth.png", np.asarray(self.img_dep * 1e3, dtype=np.uint16)
                )
            r.sleep()


if __name__ == "__main__":
    rospy.init_node("pose_estimate", anonymous=True)
    parser = argparse.ArgumentParser(description="Estimate 6D pose of given object.")
    parser.add_argument(
        "path_object", help="Path to snapshots file (typically ends with .pt)."
    )
    parser.add_argument(
        "--minL", type=int, required=True, help="Min length of matches to use."
    )
    parser.add_argument(
        "--debug", type=int, default=-1, help="Debug level, bigger means more info."
    )
    args = parser.parse_args()
    try:
        file_name = os.path.basename(args.path_object)
        obj_name = ".".join(file_name.split(".")[:-1])
        with open("./goal_pose.json", "r") as f:
            content = json.load(f)
        pose = content[obj_name]["goal_pose"]
        M_ee2m = gmatch.util.pose2mat([pose[:3], pose[3:]])
    except:
        rospy.logwarn("Failed to load goal pose, will not visualize end-effector axes.")
        M_ee2m = None

    estim = PoseEstimator(args)
    estim.reset_obj(args.path_object)
    estim.run(M_ee2m=M_ee2m)
