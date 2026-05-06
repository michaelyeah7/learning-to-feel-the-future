import os
import glob
import numpy as np
from function_util import mk_dir
from pathlib import Path
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    dataset_name: str = "dobot_usb_tac_1107"


def main(args):
    root_dir = str(Path(__file__).parent.parent.parent / "datasets/")
    dataset_dir = root_dir + "/" + args.dataset_name + "/collect_data/"
    mk_dir(dataset_dir)
    print(dataset_dir)
    output_video_dir = root_dir + "/" + args.dataset_name + "/output_videos/"
    mk_dir(output_video_dir)
    output_train_data = root_dir + "/" + args.dataset_name + "/train_data/"
    mk_dir(output_train_data)

    all_data_dir = os.listdir(dataset_dir)
    # print(all_data_dir)
    all_data_dir.sort(key=lambda x: int(x))
    # print(all_data_dir)
    max_step = 0
    for idx in range(len(all_data_dir)):
        one_data_dir = dataset_dir+all_data_dir[idx]+"/"
        data_pose_list = glob.glob(one_data_dir + 'observation/*.pkl')
        if len(data_pose_list) > max_step:
            max_step = len(data_pose_list)
    print(max_step)


if __name__ == "__main__":
    main(tyro.cli(Args))
