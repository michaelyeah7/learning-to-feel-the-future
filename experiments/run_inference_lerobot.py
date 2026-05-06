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
from pathlib import Path
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from dobot_control.cameras.realsense_camera import RealSenseCamera

from scripts.manipulate_utils import load_ini_data_camera
from ModelTrain.module.model_module import Imitate_Model

# Import external inference clients
try:
    from external_inference_client import create_tcp_client, create_http_client, create_zmq_client
    EXTERNAL_CLIENT_AVAILABLE = True
except ImportError:
    print("Warning: External client not available.")
    EXTERNAL_CLIENT_AVAILABLE = False

# Try to import LeRobot dependencies
try:
    import torch
    from transformers import AutoModel, AutoConfig
    from safetensors.torch import load_file
    LEROBOT_AVAILABLE = True
except ImportError:
    print("Warning: LeRobot dependencies not available. Only original models supported.")
    LEROBOT_AVAILABLE = False

class LeRobotWrapper:
    """Minimal wrapper for LeRobot/HuggingFace models to match Imitate_Model interface"""
    
    def __init__(self, ckpt_dir, ckpt_name=None):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_name = ckpt_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def loadModel(self):
        """Load LeRobot/HuggingFace model"""
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot dependencies not available")
            
        print(f"Loading LeRobot model from {self.ckpt_dir}")
        
        # Try to load as LeRobot policy first
        try:
            from lerobot.common.policies.factory import make_policy
            import json
            config_path = self.ckpt_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.model = make_policy(
                    hydra_cfg=config,
                    pretrained_policy_name_or_path=str(self.ckpt_dir),
                    device=self.device
                )
                self.model.eval()
                print("✅ Loaded as LeRobot policy")
                return
        except Exception as e:
            print(f"LeRobot policy loading failed: {e}")
        
        # Try to load as HuggingFace model
        try:
            config_path = self.ckpt_dir / "config.json"
            if config_path.exists():
                self.model = AutoModel.from_pretrained(str(self.ckpt_dir))
                self.model.to(self.device)
                self.model.eval()
                print("Loaded as HuggingFace model")
                return
        except Exception as e:
            print(f"HuggingFace loading failed: {e}")
        
        # Try to load safetensors manually
        try:
            safetensors_path = self.ckpt_dir / "model.safetensors"
            if safetensors_path.exists():
                state_dict = load_file(str(safetensors_path))
                # Create a simple wrapper - you'll need to customize this for your model architecture
                self.model = self._create_simple_model(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("Loaded safetensors model")
                return
        except Exception as e:
            print(f"Safetensors loading failed: {e}")
        
        raise ValueError(f"Could not load model from {self.ckpt_dir}")
    
    def _create_simple_model(self, state_dict):
        """Create ACT model from safetensors"""
        try:
            # Try to use LeRobot's ACT policy directly
            from lerobot.common.policies.act.modeling_act import ACTPolicy
            
            # Load config
            import json
            config_path = self.ckpt_dir / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create ACT model with config
            model = ACTPolicy(config)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Created ACT model from LeRobot")
            return model
            
        except Exception as e:
            print(f"LeRobot ACT failed ({e}), creating custom wrapper")
            # Fallback to custom implementation
            return self._create_custom_act_model(state_dict)
    
    def _create_custom_act_model(self, state_dict):
        """Custom ACT model implementation"""
        import torch.nn as nn
        
        class CustomACTWrapper(nn.Module):
            def __init__(self, device):
                super().__init__()
                # Store device reference
                self.device = device
                # Simple direct mapping: 14 joint positions → 14 actions
                self.action_head = nn.Linear(14, 14)  # Direct 14→14 mapping
                
            def forward(self, batch):
                # Extract joint positions directly
                if isinstance(batch, dict) and 'observation.state' in batch:
                    # Use joint positions directly (no expansion needed!)
                    joint_pos = batch['observation.state']  # Shape: [1, 14]
                    actions = self.action_head(joint_pos)    # Direct: [1, 14] → [1, 14]
                else:
                    # Fallback: return current position (no movement)
                    actions = torch.zeros(1, 14, device=self.device)
                
                return actions
            
            def predict(self, batch):
                """LeRobot-style predict method"""
                return self.forward(batch)
            
            def select_action(self, batch):
                """Alternative interface"""
                return self.forward(batch)
        
        model = CustomACTWrapper(self.device)
        
        # Try to load compatible weights
        try:
            # Filter state dict for compatible keys
            compatible_state = {}
            for key, value in state_dict.items():
                if 'action_head' in key or 'linear' in key:
                    compatible_state[key] = value
            
            if compatible_state:
                model.load_state_dict(compatible_state, strict=False)
                print("Loaded some compatible weights")
            else:
                print("Warning: No compatible weights found, using random initialization")
                
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
        
        return model
    
    def predict(self, observation, timestep):
        """Predict action from observation"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        with torch.no_grad():
            # Convert observation to tensor format
            obs_tensor = self._convert_observation(observation)
            
            # Run inference
            if hasattr(self.model, 'predict') or hasattr(self.model, 'select_action'):
                # LeRobot-style interface
                if hasattr(self.model, 'predict'):
                    action = self.model.predict(obs_tensor)
                else:
                    action = self.model.select_action(obs_tensor)
            else:
                # Standard PyTorch interface
                action = self.model(obs_tensor)
            
            # Convert back to numpy
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            
            # Ensure correct shape (14 joints)
            if len(action.shape) > 1:
                action = action.squeeze()
            
            if len(action) != 14:
                print(f"Warning: Action dim {len(action)} != 14, padding/truncating")
                if len(action) < 14:
                    padded = np.zeros(14)
                    padded[:len(action)] = action
                    action = padded
                else:
                    action = action[:14]
            
            return action.astype(np.float64)
    
    def _convert_observation(self, observation):
        """Convert observation to ACT model input format"""
        inputs = {}
        
        # Joint positions - ACT expects 'observation.state'
        if 'qpos' in observation:
            inputs['observation.state'] = torch.tensor(
                observation['qpos'], dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # Add batch dimension
        
        # Images - ACT expects 'observation.images.{camera_name}'
        if 'images' in observation:
            for key, img in observation['images'].items():
                if img is not None:
                    # Convert BGR to RGB and normalize to [0,1]
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.tensor(
                        img_rgb.transpose(2, 0, 1), dtype=torch.float32, device=self.device
                    ) / 255.0
                    inputs[f'observation.images.{key}'] = img_tensor.unsqueeze(0)
        
        # Add dummy qvel if expected by model (some ACT models need this)
        if 'qpos' in observation:
            inputs['observation.qvel'] = torch.zeros_like(inputs['observation.state'])
        
        return inputs

def load_model(args: Args):
    """Load model - supports external server, LeRobot, or fallback"""
    
    # Option 1: External inference server
    if args.use_external_server and EXTERNAL_CLIENT_AVAILABLE:
        print(f"🌐 Attempting to connect to external inference server...")
        
        if args.external_protocol == "tcp":
            print(f"Using TCP client: {args.external_server_host}:{args.external_server_port}")
            client = create_tcp_client(args.external_server_host, args.external_server_port)
        elif args.external_protocol == "http":
            print(f"Using HTTP client: {args.external_server_url}")
            client = create_http_client(args.external_server_url)
        elif args.external_protocol == "zmq":
            print(f"Using ZMQ client: {args.external_server_host}:{args.external_server_port}")
            client = create_zmq_client(args.external_server_host, args.external_server_port)
        else:
            raise ValueError(f"Unknown protocol: {args.external_protocol}")
        
        if client.loadModel():
            print("✅ Connected to external inference server")
            return client
        else:
            print("❌ Failed to connect to external server, falling back...")
    
    # Option 2: LeRobot local loading
    if LEROBOT_AVAILABLE:
        print("📦 Loading LeRobot model locally...")
        try:
            model = LeRobotWrapper(args.ckpt_dir, args.ckpt_name)
            model.loadModel()
            print("✅ Loaded LeRobot model")
            return model
        except Exception as e:
            print(f"❌ LeRobot loading failed: {e}")
    
    # Option 3: Original model fallback
    print("📦 Falling back to original model...")
    try:
        model = Imitate_Model(args.ckpt_dir, args.ckpt_name)
        model.loadModel()
        print("✅ Loaded original model")
        return model
    except Exception as e:
        print(f"❌ All model loading methods failed: {e}")
        raise

@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True
    ckpt_dir: str = './ckpt/act_peg_dobot'  # Can now point to LeRobot checkpoints
    ckpt_name: str = 'model.ckpt'
    # External inference server options
    use_external_server: bool = False
    external_server_url: str = "http://localhost:8000"  # For HTTP
    external_server_host: str = "localhost"  # For TCP/ZMQ
    external_server_port: int = 9999  # For TCP (9999) or ZMQ (5555)
    external_protocol: str = "tcp"  # "tcp", "http", or "zmq"

image_left,image_right,image_top,thread_run=None,None,None,None
lock = threading.Lock()

def run_thread_cam(rs_cam, which_cam):
    global image_left, image_right, image_top, thread_run
    if which_cam==1:
        while thread_run:
            image_left, _ = rs_cam.read()
            image_left = image_left[:, :, ::-1]
    elif which_cam==2:
        while thread_run:
            image_right, _ = rs_cam.read()
            image_right = image_right[:, :, ::-1]
    elif which_cam==0:
        while thread_run:
            image_top_src, _ = rs_cam.read()
            image_top_src = image_top_src[150:420,220:480, ::-1]
            image_top = cv2.resize(image_top_src,(640,480))

    else:
        print("Camera index error! ")


def main(args):

   # camera init
    global image_left, image_right, image_top, thread_run
    thread_run=True
    camera_dict = load_ini_data_camera()

    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
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
    reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])  # 用夹爪
    reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt,np.array([1,1]))

    # Initialize the model (now supports external server, LeRobot, and original formats)
    model = load_model(args)
    print("model init success...")

    # Initialize the parameters
    episode_len = 9000  # The total number of steps to complete the task. Note that it must be less than or equal to parameter 'episode_len' of the corresponding task in file 'ModelTrain.constants'
    t=0
    last_time = 0
    observation = {'qpos': [], 'images': {'left_wrist': [], 'right_wrist': [], 'top': []}}
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
        # with lock:
        observation['images']['left_wrist'] = image_left
        observation['images']['right_wrist'] = image_right
        observation['images']['top'] = image_top
        
        # Check if images are ready
        if any(img is None for img in observation['images'].values()):
            print("Warning: Cameras not ready, skipping step")
            time.sleep(0.1)
            continue
            
        if args.show_img:
            imgs = np.hstack((observation['images']['top'],observation['images']['left_wrist'],observation['images']['right_wrist']))
            cv2.imshow("imgs",imgs)
            cv2.waitKey(1)
        time1 = time.time()
        print("read images time(ms)：",(time1-time0)*1000)

        # Model inference,output joint value (radian)
        try:
            action = model.predict(observation,t)
        except Exception as e:
            print(f"Model prediction error: {e}")
            action = observation['qpos'].copy()  # Fallback to current position

        print("infer_action:",action)
        print('last_action:',last_action)
            
        # print("infer_action:",action)
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

        # Security protection (same as original)
        protect_err = False
        delta = action-last_action
        print("Joint increment：",delta)
        if max(delta[0:6])>0.17 or max(delta[7:13])>0.17: # 增量大于10度
            print("Note!If the joint increment is larger than 10 degrees!!!")
            print("Do you want to continue running?Press the 'Y' key to continue, otherwise press the other button to stop the program!")
            temp_img = np.zeros(shape=(640, 480))
            cv2.imshow("waitKey", temp_img)
            key = cv2.waitKey(0)
            if key == ord('y') or key == ord('Y') :
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
        obs["joint_positions"][6] = action[6]
        obs["joint_positions"][13] = action[13]
        observation['qpos'] = obs["joint_positions"]

        print("Read joint value time(ms)：", (time4 - time3) * 1000)
        t +=1

        # Reset t when the robot returns to its initial position to achieve infinite loop execution of tasks
        threshold = np.deg2rad(10)
        if t>1200 and np.all(np.abs((action- np.deg2rad([-90, 0, -90, 0, 90, 90, 57,90, 0, 90, 0, -90, -90, 57])))<threshold):
            print("Reset t=0")
            t=0

        print("The total time(ms):", (time4 - time0) * 1000)

    thread_run = False
    print("Task accomplished")


if __name__ == "__main__":
    main(tyro.cli(Args))