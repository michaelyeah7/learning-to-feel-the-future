import h5py
import numpy as np
import cv2


show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)
with h5py.File("./datasets/clean_1234/train_data/episode_init_0.hdf5", 'r',
               rdcc_nbytes=1024 ** 2 * 2) as root:
    for i in range(len(root["/observations/images/top"])):
        qpos = root["/observations/qpos"][i]
        print("step", i)
        # observation, joints angle, gripper width and images
        print("observation: left hand [J1, J2, J3, J4, J5, J6, gripper_width]:", [i for i in qpos[:7]])
        print("observation: right hand [J1, J2, J3, J4, J5, J6, gripper_width]:", [i for i in qpos[7:14]])
        show_canvas[:, :640] = np.asarray(
            cv2.imdecode(np.asarray(root["/observations/images/top"][i], dtype="uint8"), cv2.IMREAD_COLOR),
            dtype="uint8")
        show_canvas[:, 640:640 * 2] = np.asarray(
            cv2.imdecode(np.asarray(root["/observations/images/left_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR),
            dtype="uint8")
        show_canvas[:, 640 * 2:640 * 3] = np.asarray(
            cv2.imdecode(np.asarray(root["/observations/images/right_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR),
            dtype="uint8")
        cv2.imshow("0", show_canvas)

        # predict joint angle, gripper width
        action = root["action"][i]
        print("predict action: left hand [J1, J2, J3, J4, J5, J6, gripper_width]:", [i for i in action[:7]])
        print("predict action: right hand [J1, J2, J3, J4, J5, J6, gripper_width]:", [i for i in action[7:14]])
        print()


        cv2.waitKey(0)