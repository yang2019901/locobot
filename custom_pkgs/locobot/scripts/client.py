import socket
import pickle
import struct
import numpy as np
import os
from typing import Type
from PIL import Image
import sys
import cv2
import traceback

from data_models import *

class ServiceProxy:
    """
    A proxy for a remote service that is similar to ROS ServiceProxy.
    It is initialized with request and response classes and can be called
    with the arguments of the request class.
    """
    def __init__(self, request_class: Type, response_class: Type, host, port):
        self.request_class = request_class
        self.response_class = response_class
        self.host = host
        self.port = port

    def __call__(self, **kwargs):
        request = self.request_class(**kwargs)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((self.host, self.port))

            data = pickle.dumps(request)
            message = struct.pack("Q", len(data)) + data
            client_socket.sendall(message)
            print("Request sent to server.")

            response_data = b""
            payload_size = struct.calcsize("Q")

            while len(response_data) < payload_size:
                packet = client_socket.recv(4 * 1024)
                if not packet:
                    raise ConnectionError("Socket connection broken")
                response_data += packet
            
            packed_msg_size = response_data[:payload_size]
            response_data = response_data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(response_data) < msg_size:
                response_data += client_socket.recv(4 * 1024)
            
            frame_data = response_data[:msg_size]
            
            response = pickle.loads(frame_data)
            print("Response received from server.")

            if not isinstance(response, self.response_class):
                raise TypeError(f"Expected response of type {self.response_class.__name__}, but got {type(response).__name__}")

            return response

def get_point_cloud(data_dir, label):
    # get data
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    mask = np.array(Image.open(os.path.join(data_dir, f'mask_{label}.png')))
    
    # get camera intrinsics
    with open(os.path.join(data_dir, 'cam_intrin.txt'), 'r') as f:
        l1, l2 = f.readlines()[:2]
        l1 = list(map(float, l1.strip().split()))
        l2 = list(map(float, l2.strip().split()))
        fx, cx = l1[0], l1[2]
        fy, cy = l2[1], l2[2]
    scale = 1000.0

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (mask > 0)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    
    points_masked = points[mask].astype(np.float32)
    colors_masked = colors[mask].astype(np.float32)

    rgbcld = np.concatenate([points_masked, colors_masked], axis=-1)
    
    return np.round(rgbcld, 4)

def test_anygrasp(host, port):
    data_dir = '/home/yang2019901/grasp_data'
    label = 'bowl'
    
    try:
        print(f"\nPreparing point cloud for label: '{label}'")
        point_cloud = get_point_cloud(data_dir, label)
        print(f"Point cloud shape: {point_cloud.shape}")

        # Create a service proxy for the AnyGrasp service
        anygrasp_proxy = ServiceProxy(AnygraspRequest, AnygraspResponse, host, port)
        
        # Call the service by passing the point cloud
        response = anygrasp_proxy(points=point_cloud.tolist())

        print("\n--- Results ---")
        if not response.grasps:
            print("No grasps detected.")
        else:
            print(f"Number of grasps detected: {len(response.grasps)}")
            best_grasp = response.grasps[0]
            print("\nBest Grasp (4x4 Matrix):")
            print(best_grasp)
            
            # # Save the best grasp to a file
            # grasp_filename = "best_grasp.txt"
            # np.savetxt(grasp_filename, best_grasp, fmt='%.4f')
            # print(f"\nSaved the best grasp to '{grasp_filename}'")

    except Exception as e:
        print(f"\nAn error occurred during the client call: {e}")
        traceback.print_exc()

def test_gsam(host, port):
    img = cv2.imread('/home/yang2019901/grasp_data/color.png', cv2.IMREAD_COLOR_RGB)
    print(img.shape)
    prompt = "bowl. handle."

    try:
        print(f"\nCalling GSAM service with prompt: '{prompt}'")
            
        # Create a service proxy for the GSAM service
        gsam_proxy = ServiceProxy(GsamRequest, GsamResponse, host, port)
            
        # Call the service by passing arguments for the GsamRequest
        response:GsamResponse = gsam_proxy(image=img.tolist(), prompt=prompt)

        print("\n--- Results ---")
        print(f"Number of objects detected: {len(response.masks)}")
            
        for i, (mask, label, conf) in enumerate(zip(response.masks, response.labels, response.confidences)):
            mask = np.array(mask, dtype=np.uint8)
            print(f"\nObject {i+1}:")
            print(f"  Label: {label}")
            print(f"  Confidence: {conf:.4f}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask dtype: {mask.dtype}")
                
            # Save the mask as an image file
            mask_filename = f"result_mask_{i}_{label}.png"
            # Convert boolean mask to 0-255 image for saving
            cv2.imwrite(mask_filename, mask)
            print(f"  Saved mask to '{mask_filename}'")

    except Exception as e:
        print(f"\nAn error occurred during the client call: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    # test_gsam('172.27.80.1', 8001)
    test_anygrasp('172.27.80.1', 8002)