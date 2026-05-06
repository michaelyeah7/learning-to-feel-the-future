import numpy as np, torch, os, h5py, fnmatch, cv2
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import IPython
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in iter(sublist)]


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class, use_vitg=False, tactile_camera_names=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.tactile_camera_names = tactile_camera_names if tactile_camera_names else []
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.use_vitg = use_vitg
        if self.policy_class == "Diffusion":
            self.augment_images = True
        else:
            self.augment_images = False
        self.transformations = None
        self.__getitem__(0)
        self.is_sim = False

    def __len__(self):
        return len(self.episode_ids)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return (episode_id, start_ts)

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        with h5py.File(dataset_path, "r") as root:
            try:
                is_sim = root.attrs["sim"]
            except:
                is_sim = False

            compressed = root.attrs.get("compress", False)
            if "/base_action" in root:
                base_action = root["/base_action"][()]
                base_action = preprocess_base_action(base_action)
                action = np.concatenate([root["/action"][()], base_action], axis=(-1))
            else:
                action = root["/action"][()]
                dummy_base_action = np.zeros([action.shape[0], 2])
                action = np.concatenate([action, dummy_base_action], axis=(-1))
            original_action_shape = action.shape
            episode_len = original_action_shape[0]
            qpos = root["/observations/qpos"][start_ts]
            qvel = root["/observations/qvel"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                # Try /observations/images/{cam_name} first (RGB cameras)
                # Then try /observations/{cam_name} (tactile sensors)
                if f"/observations/images/{cam_name}" in root:
                    image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
                elif f"/observations/{cam_name}" in root:
                    image_dict[cam_name] = root[f"/observations/{cam_name}"][start_ts]
                else:
                    raise KeyError(f"Cannot find {cam_name} in /observations/images/ or /observations/")

            if compressed:
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)

            if is_sim:
                action = action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action[max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)
        padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=(np.float32))
        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_episode_len)
        is_pad[action_len:] = 1
        padded_action = padded_action[:self.chunk_size]
        is_pad = is_pad[:self.chunk_size]
        
        # Separate RGB cameras from tactile sensors
        rgb_cameras = [cam for cam in self.camera_names if cam not in self.tactile_camera_names]
        
        # Process RGB camera images (stack together)
        rgb_images = []
        for cam_name in rgb_cameras:
            img = image_dict[cam_name]  # (H, W, C)
            rgb_images.append(img)
        
        if rgb_images:
            # Stack RGB images and convert to tensor
            rgb_stacked = np.stack(rgb_images, axis=0)  # (num_rgb, H, W, C)
            rgb_tensor = torch.from_numpy(rgb_stacked)
            rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)  # (num_rgb, C, H, W)
            rgb_tensor = rgb_tensor / 255.0  # Normalize
        else:
            rgb_tensor = None
        
        # Process tactile images (keep separate, resize for ViTG)
        tactile_images = []
        for cam_name in self.tactile_camera_names:
            img = image_dict[cam_name]  # (H, W, C)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img).float()
            img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
            img_tensor = img_tensor / 255.0  # Normalize
            
            # Resize to 224x224 for ViTG
            if self.use_vitg:
                resize_transform = transforms.Resize((224, 224), antialias=True)
                img_tensor = resize_transform(img_tensor)
            
            tactile_images.append(img_tensor)
        
        # Prepare image_data (RGB only, already stacked)
        if rgb_tensor is not None:
            image_data = rgb_tensor  # (num_rgb, C, H, W)
        else:
            # No RGB cameras, create empty tensor
            image_data = torch.empty(0, 3, 480, 640)
        
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        
        # Apply augmentation if needed (only to RGB, not tactile)
        if self.augment_images and image_data.shape[0] > 0:
            if self.transformations is None:
                print("Initializing transformations")
                self.transformations = [
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)]
            
            # Apply to RGB images
            for transform in self.transformations:
                image_data = transform(image_data)

        if self.policy_class == "Diffusion":
            action_data = (action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"]) * 2 - 1
        else:
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return (image_data, tactile_images, qpos_data, action_data, is_pad)


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []
    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                qvel = root["/observations/qvel"][()]
                if "/base_action" in root:
                    base_action = root["/base_action"][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root["/action"][()], base_action], axis=(-1))
                else:
                    action = root["/action"][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=(-1))
        except Exception as e:
            try:
                print(f"Error loading {dataset_path} in get_norm_stats")
                print(e)
                quit()
            finally:
                e = None
                del e

        else:
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
            all_episode_len.append(len(qpos))
    else:
        all_qpos_data = torch.cat(all_qpos_data, dim=0)
        all_action_data = torch.cat(all_action_data, dim=0)
        action_mean = all_action_data.mean(dim=[0]).float()
        action_std = all_action_data.std(dim=[0]).float()
        action_std = torch.clip(action_std, 0.01, np.inf)
        qpos_mean = all_qpos_data.mean(dim=[0]).float()
        qpos_std = all_qpos_data.std(dim=[0]).float()
        qpos_std = torch.clip(qpos_std, 0.01, np.inf)
        action_min = all_action_data.min(dim=0).values.float()
        action_max = all_action_data.max(dim=0).values.float()
        eps = 0.0001
        stats = {'action_mean':(action_mean.numpy)(),  'action_std':(action_std.numpy)(),  'action_min':(action_min.numpy()) - eps, 
         'action_max':(action_max.numpy()) + eps,  'qpos_mean':(qpos_mean.numpy)(), 
         'qpos_std':(qpos_std.numpy)(),  'example_qpos':qpos}
        return (
         stats, all_episode_len)


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            if "features" in filename:
                pass
            elif skip_mirrored_data and "mirror" in filename:
                pass
            else:
                hdf5_files.append(os.path.join(root, filename))
        else:
            print(f"Found {len(hdf5_files)} hdf5 files")
            return hdf5_files


def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice((len(episode_len_l)), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        else:
            yield batch


def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99, use_vitg=False, tactile_camera_names=None):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [
         dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f"\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n")
    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    else:
        if type(stats_dir_l) == str:
            stats_dir_l = [
             stats_dir_l]
    
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f"Norm stats from: {stats_dir_l}")
    if use_vitg:
        print("Dataset configured for ViTG: tactile images will be resized to 224x224")
    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class, use_vitg=use_vitg, tactile_camera_names=tactile_camera_names)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class, use_vitg=use_vitg, tactile_camera_names=tactile_camera_names)
    train_num_workers = (8 if os.getlogin() == "zfu" else 16) if train_dataset.augment_images else 2
    val_num_workers = 8 if train_dataset.augment_images else 2
    print(f"Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}")
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=2)
    return (
     train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim)


def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0
    v = base_action[(Ellipsis, 0)]
    w = base_action[(Ellipsis, 1)]
    base_action = base_action.copy()
    base_action[(Ellipsis, 0)] = v - c * w
    return base_action


def smooth_base_action(base_action):
    return np.stack([np.convolve(base_action[:, i], np.ones(5) / 5, mode="same") for i in range(base_action.shape[1])],
      axis=(-1)).astype(np.float32)


def preprocess_base_action(base_action):
    base_action = smooth_base_action(base_action)
    return base_action


def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    return np.array([linear_vel, angular_vel])


def sample_box_pose():
    x_range = [
     0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    x_range = [
     0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])
    x_range = [
     -0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])
    return (
     peg_pose, socket_pose)


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        else:
            result[k] = value_sum / num_items

    else:
        return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    else:
        return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# okay decompiling utils.pyc
