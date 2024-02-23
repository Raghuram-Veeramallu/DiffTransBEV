import cv2
import numpy as np

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from constants.data_constants import CAMERA_SENSOR_LIST, LIDAR_SENSOR, VALID_DATA_VERSIONS
from models.view_transformer.lss_utils import gen_dx_bx, get_grid_config, get_normalization_ops


class NuScenesDataLoader():
    def __init__(self, config):
        self.num_replicas = config.world_size
        self.rank = config.rank
        self.distributed = config.distributed_run
        self.data_workers = config.data.data_workers

    def get_dataloader(self, config, stage, batch_size):
        dataset = self.get_dataset(config)
        if self.distributed and stage in ['train']:
            self.train_sampler = DistributedSampler(
                dataset, num_replicas=self.num_replicas, rank=self.rank)
        else:
            self.train_sampler = None

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None and stage not in ('val', 'test')),
            num_workers=self.data_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True
        )
        return data_loader

    def get_dataset(self, config):
        return NuScenesDataset(config)

    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

class NuScenesDataset(Dataset):

    def __init__(self, config):
        version = config.data.version
        dataroot = config.data.rootpath
        is_train = config.data.is_train

        assert version in VALID_DATA_VERSIONS
        # setting verbose true to condense the logs for better debugging
        self.nuscenes = NuScenes(version, dataroot, verbose=False)

        # get scenes names for train and validation
        self.scene_names = self.get_splits(version, is_train)
        self.samples = self.get_samples_based_on_scenes(self.scene_names)

        grid_config = get_grid_config(config)
        # grid_config = {
        #     'xbound': [-50.0, 50.0, 0.5],
        #     'ybound': [-50.0, 50.0, 0.5],
        #     'zbound': [-10.0, 10.0, 20.0],
        #     'dbound': [4.0, 45.0, 1.0],
        # }
        dx, bx, nx = gen_dx_bx(grid_config['xbound'], grid_config['ybound'], grid_config['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        self.normalize_img = get_normalization_ops()

    def get_splits(self, version, is_train):
        """
        Get the splits for training and validation
        """
        # using the splits module provided by NuScenes python SDK
        if version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise ValueError('unknown')

        if is_train:
            return train_scenes
        else:
            return val_scenes

    def get_samples_based_on_scenes(self, scenes):
        samples = [samp for samp in self.nuscenes.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nuscenes.get('scene', samp['scene_token'])['name'] in scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs = []
        rots = []
        trans = []
        intrins = []
        for each_sensor in CAMERA_SENSOR_LIST:
            sensor_sample = sample['data'][each_sensor]
            sample_data = self.nuscenes.get('sample_data', sensor_sample)
            image_path, _, _ = self.nuscenes.get_sample_data(sample_data['token'])
            image = Image.open(image_path).convert('RGB')

            sens = self.nuscenes.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            imgs.append(self.normalize_img(image))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)

        egopose = self.nuscenes.get('ego_pose', self.nuscenes.get('sample_data', sample['data'][LIDAR_SENSOR])['ego_pose_token'])

        ann_img = np.zeros((int(self.nx[0]), int(self.nx[1])))

        transl = -np.array(egopose['translation'])
        rotat = Quaternion(egopose['rotation']).inverse
        # img = np.zeros((self.nx[0]))
        for each_ann in sample['anns']:
            instance = self.nuscenes.get('sample_annotation', each_ann)
            box = Box(instance['translation'], instance['size'], Quaternion(instance['rotation']))
            box.translate(transl)
            box.rotate(rotat)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(ann_img, [pts], 1.0)

        # we are passing a binary image representing the precense of agents surrounding the ego vehicle
        return torch.stack(imgs), torch.stack(rots), torch.stack(trans), torch.stack(intrins), torch.Tensor(ann_img).unsqueeze(0)
