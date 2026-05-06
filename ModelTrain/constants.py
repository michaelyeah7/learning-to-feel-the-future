### Task parameters
import pathlib
import os

# Override with `export DOBOT_DATA_DIR=/path/to/datasets` or edit this default.
DATA_DIR = os.environ.get(
    "DOBOT_DATA_DIR",
    str(pathlib.Path(__file__).resolve().parents[1] / "datasets"),
)
TASK_CONFIGS = {
    # dobot clean dishes
    'clean_dishes_task': {
        'dataset_dir': DATA_DIR + '/train_data',
        'episode_len': 1000,  # Set to 1200 during training and 10000 during inference
        'train_ratio': 0.98,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },


    # dobot move cube new
    'move_cube_new': {
        'dataset_dir': DATA_DIR + '/dataset_package_test/train_data/',
        'episode_len': 4800,
        'train_ratio': 0.98,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },


    # dobot floder closh
    'floder_closh': {
        'dataset_dir': DATA_DIR + '/floder_closh',
        'episode_len': 2000,  # 1100,  # 900,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'floder_closh_cotrain': {
        'dataset_dir': [
            DATA_DIR + '/floader_closh',
            DATA_DIR + '/clean_disk5',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/floder_closh',
        ],
        'sample_weights': [5, 5],
        'train_ratio': 0.9,  # ratio of train data from the first dataset_dir
        'episode_len': 2000,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    
    'dobot_peg_in_hole': {
        'dataset_dir': DATA_DIR + '/peginhole/train_data',
        'episode_len': 900,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    # This task is meant to only pick from random location using left arm and pass to right arm.
    'dobot_pick_random': {
        'dataset_dir': DATA_DIR + '/dobot_pick_random_1005/train_data',
        'episode_len': 800,
        'train_ratio': 0.96,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'dobot_pick_random_1008': {
        'dataset_dir': DATA_DIR + '/dobot_pick_random_1008/train_data',
        'episode_len': 800,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'dobot_pick_random_1010': {
        'dataset_dir': DATA_DIR + '/dobot_pick_random_1010/train_data',
        'episode_len': 700,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'dobot_pick_random_1010_prof': {
        'dataset_dir': DATA_DIR + '/dobot_pick_random_1010_prof/train_data',
        'episode_len': 900,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'dobot_pick_random_1011': {
        'dataset_dir': DATA_DIR + '/dobot_pick_random_1011/train_data',
        'episode_len': 800,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'dobot_pick_random_1013': {
        'dataset_dir': DATA_DIR + '/dobot_pick_random_1013/train_data',
        'episode_len': 800,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },


    # This task is meant to only pick and peg from the base plate.
    'dobot_peg_random': {
        'dataset_dir': DATA_DIR + '/dobot_peg_random_1005/train_data',
        'episode_len': 550,
        'train_ratio': 0.96,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    # This task is the mixture of 'dobot_pick_random' and 'dobot_peg_random'.
    'dobot_pick_peg_random': {
        'dataset_dir': DATA_DIR + '/dobot_pick_peg_random_1005/train_data',
        'episode_len': 800,
        'train_ratio': 0.98,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'dobot_peg_fixed_tactile': {
        'dataset_dir': DATA_DIR + '/dobot_peg_fixed_tactile/train_data',
        'num_episodes': 50,
        'episode_len': 700,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'dobot_peginhole_tac_1029': {
        'dataset_dir': DATA_DIR + '/dobot_peginhole_tac_1029/train_data',
        'episode_len': 700,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist'],  # RGB cameras
        'tactile_camera_names': ['tactile1']  # Tactile sensors for ViTG
    },
    'dobot_peginhole_tac_1107': {
        'dataset_dir': DATA_DIR + '/dobot_peginhole_tac_1107/train_data',
        'episode_len': 350,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist'],  # RGB cameras
        'tactile_camera_names': ['tactile1']  # Tactile sensors for ViTG
    },
    'dobot_gearassemb_tac_1107': {
        'dataset_dir': DATA_DIR + '/dobot_gearassemb_tac_1107/train_data',
        'episode_len': 700,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist'],  # RGB cameras
        'tactile_camera_names': ['tactile1']  # Tactile sensors for ViTG
    },
    'dobot_usb_tac_1107': {
        'dataset_dir': DATA_DIR + '/dobot_usb_tac_1107/train_data',
        'episode_len': 500,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist'],  # RGB cameras
        'tactile_camera_names': ['tactile1']  # Tactile sensors for ViTG
    },
}

###  fixed constants
DT = 0.02
FPS = 50




