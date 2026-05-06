"""
Inference script for ACTJEPAAdapter policy
Supports ACTJEPA and ACTJEPAAdapter policies with tactile sensors
"""
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+"/ModelTrain/")
import cv2
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
import queue
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from dobot_control.cameras.realsense_camera import RealSenseCamera

from scripts.manipulate_utils import load_ini_data_camera

from ModelTrain.module.model_module import Imitate_Model
from digit_interface import Digit


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True
    ckpt_dir: str = "./ckpt/actjepa_adapter_vitl"
    ckpt_name: str = "policy_last.ckpt"
    policy_class: str = "ACTJEPAAdapter"  # or "ACTJEPA" for baseline
    task_name: str = "dobot_peginhole_tac_1029"  # Task name for loading config
    vit_ckpt_path: str = "./jepa_ckpt/vitl.pt"  # Path to ViT checkpoint
    vit_model: str = "vitl"  # ViT model type: "vitl" (1024-dim) or "vitg" (1408-dim)
    temporal_agg: bool = True  # Use temporal aggregation for smooth actions

image_left,image_right,image_top,image_tactile,thread_run=None,None,None,None,None
lock = threading.Lock()

def run_thread_cam(rs_cam, which_cam):
    global image_left, image_right, image_top, image_tactile, thread_run
    if which_cam==1:  # left wrist
        while thread_run:
            image_left, _ = rs_cam.read()
            image_left = image_left[:, :, ::-1]
    elif which_cam==2:  # right wrist
        while thread_run:
            image_right, _ = rs_cam.read()
            image_right = image_right[:, :, ::-1]
    elif which_cam==0:  # top
        while thread_run:
            image_top_src, _ = rs_cam.read()
            image_top_src = image_top_src[150:420,220:480, ::-1]
            image_top = cv2.resize(image_top_src,(640,480))
    elif which_cam==3:  # tactile sensor
        while thread_run:
            image_tactile_src, _ = rs_cam.read()
            # Resize tactile image to 224x224 for ViT
            image_tactile = cv2.resize(image_tactile_src[:, :, ::-1], (224, 224))
    else:
        print("Camera index error! ")


def main(args):

   # camera init
    global image_left, image_right, image_top, image_tactile, thread_run
    thread_run=True
    camera_dict = load_ini_data_camera()

    print(f"Initializing cameras for policy: {args.policy_class}")
    
    # RGB cameras
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    
    # Tactile camera (if available)
    tactile1 = Digit('D21168')
    tactile1.connect()
    print("tactile1 connected")
    
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    
    thread_cam_left.start()
    thread_cam_right.start()
    thread_cam_top.start()
    
    show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)
    time.sleep(2)
    print("camera thread init success...")

   # robot init
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    print("robot init success...")

    # go to the safe position
    reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
    reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt,np.array([1,1]))
    time.sleep(1)

    # go to the initial photo position
    reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])  # with gripper
    reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt,np.array([1,1]))

    # Initialize the model
    print(f"Loading model from: {args.ckpt_dir}/{args.ckpt_name}")
    print(f"Policy class: {args.policy_class}")
    model = Imitate_Model(ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name)
    model.loadModel()
    print("model init success...")

    # Initialize the parameters
    episode_len = 90000  # The total number of steps to complete the task
    t=0
    last_time = 0
    observation = {'qpos': [], 'images': {'left_wrist': [], 'right_wrist': [], 'top': []}}
    
    # Add tactile sensor to observation if using JEPA policies
    if args.policy_class in ["ACTJEPA", "ACTJEPAAdapter"]:
        observation['tactile1'] = []  # Tactile sensor data
    
    obs = env.get_obs()
    obs["joint_positions"][6] = 1.0  # Initial position of the gripper
    obs["joint_positions"][13] = 1.0
    observation['qpos'] = obs["joint_positions"]  # Initial value of the joint
    last_action = observation['qpos'].copy()

    first = True

    print("The robot begins to perform tasks autonomously...")
    while t < episode_len:
        # Obtain the current images
        time0 = time.time()
        observation['images']['left_wrist'] = image_left
        observation['images']['right_wrist'] = image_right
        observation['images']['top'] = image_top
        
        # Add tactile sensor data
        if args.policy_class in ["ACTJEPA", "ACTJEPAAdapter"]:
            tactile_frame = tactile1.get_frame()
            observation['tactile1'] = cv2.resize(tactile_frame, (224, 224))
        
        if args.show_img:
            # Show RGB images
            imgs = np.hstack((observation['images']['top'],
                             observation['images']['left_wrist'],
                             observation['images']['right_wrist']))
            
            # Show tactile if available
            if args.policy_class in ["ACTJEPA", "ACTJEPAAdapter"] and observation.get('tactile1') is not None:
                tactile_resized = cv2.resize(observation['tactile1'], (640, 480))
                imgs = np.hstack((imgs, tactile_resized))
            
            cv2.imshow("imgs",imgs)
            cv2.waitKey(1)
        time1 = time.time()
        print("read images time(ms)：",(time1-time0)*1000)

        # Model inference,output joint value (radian)
        action = model.predict(observation,t)
        # Clip gripper values
        if action[6]>1:
            action[6]=1
        elif action[6]<0:
            action[6] = 0
        if action[13]>1:
            action[13]=1
        elif action[13]<0:
            action[13]=0
        time2 = time.time()
        print("Model inference time(ms)：", (time2 - time1) * 1000)

        # Security protection
        protect_err = False

        delta = action-last_action
        print("Joint increment：",delta)
        if max(delta[0:6])>0.17 or max(delta[7:13])>0.17: # increment larger than 10 degrees
            print("Note!If the joint increment is larger than 10 degrees!!!")
            print("Do you want to continue running?Press the 'Y' key to continue, otherwise press the other button to stop the program!")
            temp_img = np.zeros(shape=(640, 480))
            cv2.imshow("waitKey", temp_img)
            key = cv2.waitKey(0)
            if key == ord('y') or key == ord('Y'):
                cv2.destroyWindow("waitKey")
                max_delta = (np.abs(last_action - action)).max()
                steps = min(int(max_delta / 0.001), 100)
                for jnt in np.linspace(last_action, action, steps):
                    env.step(jnt,np.array([1,1]))
                first = False
            else:
                protect_err = True
                cv2.destroyAllWindows()

        # Joint angle limitations
        if not ((action[2] > -2.6 and action[2] < 0 and action[3] > -0.6) and \
                (action[9] < 2.6 and action[9] > 0 and action[10] < 0.6)):
            print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")
            print(action)
            protect_err = True

        # Position limits
        t1 = time.time()
        pos = env.get_XYZrxryrz_state()
        if not ((pos[0] > -410 and pos[0] < 300 and pos[1] > -700 and pos[1] < -210 and pos[2] > 42) and \
                (pos[6] < 410 and pos[6] > -250 and pos[7] > -700 and pos[7] < -210 and pos[8] > 42)):
            print("[Warn]:The robot arm XYZ is out of the safe position! ")
            print(pos)
            protect_err = True
        t2 = time.time()
        print("get pos time(ms):", (t2 - t1)* 1000)

        if protect_err:
            env.set_do_status([3, 0])  # yellow light off
            env.set_do_status([2, 0])  # green light off
            env.set_do_status([1, 1])  # red light on
            time.sleep(1)
            exit()

        if first:
            max_delta = (np.abs(last_action - action)).max()
            steps = min(int(max_delta / 0.001), 100)
            for jnt in np.linspace(last_action, action, steps):
                env.step(jnt,np.array([1,1]))
            first = False

        last_action = action.copy()

        # Control robot movement
        time3 = time.time()
        obs = env.step(action,np.array([1,1]))
        time4 = time.time()

        # Obtain the current joint value of the robots (including the gripper)
        obs["joint_positions"][6] = action[6]   # Use last action for gripper
        obs["joint_positions"][13] = action[13]
        observation['qpos'] = obs["joint_positions"]

        print("Read joint value time(ms)：", (time4 - time3) * 1000)
        t +=1

        # Reset t when the robot returns to its initial position
        threshold = np.deg2rad(10)
        if t>1200 and np.all(np.abs((action- np.deg2rad([-90, 0, -90, 0, 90, 90, 57,90, 0, 90, 0, -90, -90, 57])))<threshold):
            print("Reset t=0")
            t=0

        print("The total time(ms):", (time4 - time0) * 1000)


    thread_run = False
    print("Task accomplished")


if __name__ == "__main__":
    main(tyro.cli(Args))

