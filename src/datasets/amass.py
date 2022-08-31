# # -*- coding: utf-8 -*-
#
# # Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# # holder of all proprietary rights on this computer program.
# # You can only use this computer program if you have closed
# # a license agreement with MPG or you get the right to use the computer
# # program from someone who is authorized to grant you that right.
# # Any use of the computer program without a valid license is prohibited and
# # liable to prosecution.
# #
# # Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# # der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# # for Intelligent Systems. All rights reserved.
# #
# # Contact: ps-license@tuebingen.mpg.de
#
# import torch
# import joblib
# import numpy as np
# import os.path as osp
# from torch.utils.data import Dataset
#
# # from lib.core.config import VIBE_DB_DIR
# VIBE_DB_DIR = '../VIBE/data/vibe_db'
# # from lib.data_utils.img_utils import split_into_chunks
#
# def split_into_chunks(vid_names, seqlen, stride):
#     video_start_end_indices = []
#
#     video_names, group = np.unique(vid_names, return_index=True)
#     perm = np.argsort(group)
#     video_names, group = video_names[perm], group[perm]
#
#     indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])
#
#     for idx in range(len(video_names)):
#         indexes = indices[idx]
#         if indexes.shape[0] < seqlen:
#             continue
#         chunks = view_as_windows(indexes, (seqlen,), step=stride)
#         start_finish = chunks[:, (0, -1)].tolist()
#         video_start_end_indices += start_finish
#
#     return video_start_end_indices
#
# class AMASS(Dataset):
#     def __init__(self, seqlen):
#         self.seqlen = seqlen
#
#         self.stride = seqlen
#
#         self.db = self.load_db()
#         self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
#         del self.db['vid_name']
#         print(f'AMASS dataset number of videos: {len(self.vid_indices)}')
#
#     def __len__(self):
#         return len(self.vid_indices)
#
#     def __getitem__(self, index):
#         return self.get_single_item(index)
#
#     def load_db(self):
#         db_file = osp.join(VIBE_DB_DIR, 'amass_db.pt')
#         db = joblib.load(db_file)
#         return db
#
#     def get_single_item(self, index):
#         start_index, end_index = self.vid_indices[index]
#         thetas = self.db['theta'][start_index:end_index+1]
#
#         cam = np.array([1., 0., 0.])[None, ...]
#         cam = np.repeat(cam, thetas.shape[0], axis=0)
#         theta = np.concatenate([cam, thetas], axis=-1)
#
#         target = {
#             'theta': torch.from_numpy(theta).float(),  # cam, pose and shape
#         }
#         return target

import os
import numpy as np
import joblib
from .dataset import Dataset
from src.config import ROT_CONVENTION_TO_ROT_NUMBER
from src import config
from PIL import Image
import sys

sys.path.append('')

# action2motion_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 24, 38]
# change 0 and 8
action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]  # [18,]

from src.utils.action_label_to_idx import action_label_to_idx, idx_to_action_label


def get_z(cam_s, cam_pos, joints, img_size, flength):
    """
    Solves for the depth offset of the model to approx. orth with persp camera.
    """
    # Translate the model itself: Solve the best z that maps to orth_proj points
    joints_orth_target = (cam_s * (joints[:, :2] + cam_pos) + 1) * 0.5 * img_size
    height3d = np.linalg.norm(np.max(joints[:, :2], axis=0) - np.min(joints[:, :2], axis=0))
    height2d = np.linalg.norm(np.max(joints_orth_target, axis=0) - np.min(joints_orth_target, axis=0))
    tz = np.array(flength * (height3d / height2d))
    return float(tz)


def get_trans_from_vibe(vibe, use_z=True):
    alltrans = []
    for t in range(vibe["joints3d"].shape[0]):
        # Convert crop cam to orig cam
        # No need! Because `convert_crop_cam_to_orig_img` from demoutils of vibe
        # does this already for us :)
        # Its format is: [sx, sy, tx, ty]
        cam_orig = vibe["orig_cam"][t]
        x = cam_orig[2]
        y = cam_orig[3]
        if use_z:
            z = get_z(cam_s=cam_orig[0],  # TODO: There are two scales instead of 1.
                      cam_pos=cam_orig[2:4],
                      joints=vibe['joints3d'][t],
                      img_size=480,
                      flength=500)
            # z = 500 / (0.5 * 480 * cam_orig[0])
        else:
            z = 0
        trans = [x, y, z]
        alltrans.append(trans)
    alltrans = np.array(alltrans)
    return alltrans - alltrans[0]


class AMASS(Dataset):
    dataname = "amass"

    def __init__(self, datapath="data/amass/amass_30fps_legacy_db.pt", split="train", use_z=1, **kwargs):
        assert '_db.pt' in datapath
        self.datapath = datapath.replace('_db.pt', '_{}.pt'.format(split))
        assert os.path.exists(self.datapath)
        print('datapath used by amass is [{}]'.format(self.datapath))
        super().__init__(**kwargs)

        self.dataname = "amass"

        # FIXME - hardcoded:
        self.rot_convention = 'legacy'
        self.use_betas = False
        self.use_gender = False
        self.use_body_features = False
        if 'clip_preprocess' in kwargs.keys():
            self.clip_preprocess = kwargs['clip_preprocess']

        self.use_z = (use_z != 0)

        # keep_actions = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
        dummy_class = [0]
        genders = config.GENDERS
        self.num_classes = len(dummy_class)

        self.db = self.load_db()
        self._joints3d = []
        self._poses = []
        self._num_frames_in_video = []
        self._actions = []
        self._betas = []
        self._genders = []
        self._heights = []
        self._masses = []
        self._clip_images = []
        self._clip_texts = []
        self._clip_pathes = []
        self._actions_cat = []
        self.clip_label_text = "text_raw_labels"  # "text_proc_labels"

        seq_len = 100
        n_sequences = len(self.db['thetas'])
        # split sequences
        for seq_idx in range(n_sequences):
            n_sub_seq = self.db['thetas'][seq_idx].shape[0] // seq_len
            if n_sub_seq == 0: continue
            n_frames_in_use = n_sub_seq * seq_len
            joints3d = np.split(self.db['joints3d'][seq_idx][:n_frames_in_use], n_sub_seq)
            poses = np.split(self.db['thetas'][seq_idx][:n_frames_in_use], n_sub_seq)
            self._joints3d.extend(joints3d)
            self._poses.extend(poses)
            self._num_frames_in_video.extend([seq_len] * n_sub_seq)

            if 'action_cat' in self.db:
                self._actions_cat.extend(np.split(self.db['action_cat'][seq_idx][:n_frames_in_use], n_sub_seq))

            if self.use_betas:
                self._betas.extend(np.split(self.db['betas'][seq_idx][:n_frames_in_use], n_sub_seq))
            if self.use_gender:
                self._genders.extend([str(self.db['genders'][seq_idx]).replace("b'female'", "female").replace("b'male'",
                                                                                                              "male")] * n_sub_seq)
            if self.use_body_features:
                self._heights.extend([self.db['heights'][seq_idx]] * n_sub_seq)
                self._masses.extend([self.db['masses'][seq_idx]] * n_sub_seq)
            if 'clip_images' in self.db.keys():
                images = [np.squeeze(e) for e in np.split(self.db['clip_images'][seq_idx][:n_sub_seq], n_sub_seq)]
                processed_images = [self.clip_preprocess(Image.fromarray(img)) for img in images]
                self._clip_images.extend(processed_images)
            if self.clip_label_text in self.db:
                self._clip_texts.extend(np.split(self.db[self.clip_label_text][seq_idx][:n_frames_in_use], n_sub_seq))
            if 'clip_pathes' in self.db:
                self._clip_pathes.extend(np.split(self.db['clip_pathes'][seq_idx][:n_sub_seq], n_sub_seq))
            if 'clip_images_emb' in self.db.keys():
                self._clip_images_emb.extend(np.split(self.db['clip_images_emb'][seq_idx][:n_sub_seq], n_sub_seq))



            actions = [0] * n_sub_seq
            self._actions.extend(actions)

        assert len(self._num_frames_in_video) == len(self._poses) == len(self._joints3d) == len(self._actions)
        if self.use_betas:
            assert len(self._poses) == len(self._betas)
        if self.use_gender:
            assert len(self._poses) == len(self._genders)
        if 'clip_images' in self.db.keys():
            assert len(self._poses) == len(self._clip_images)

        self._actions = np.array(self._actions)
        self._num_frames_in_video = np.array(self._num_frames_in_video)

        N = len(self._poses)
        # same set for training and testing
        self._train = np.arange(N)
        self._test = np.arange(N)

        self._action_to_label = {x: i for i, x in enumerate(dummy_class)}
        self._label_to_action = {i: x for i, x in enumerate(dummy_class)}

        self._gender_to_label = {x: i for i, x in enumerate(genders)}
        self._label_to_gender = {i: x for i, x in enumerate(genders)}

        self._action_classes = idx_to_action_label

    def load_db(self):
        # Load amass dataset encoded to a .db file
        # The loaded data is structured:
        # {
        #     'theta': [data_size, 82] (float64) (structured [pose(72), betas(10)])
        #     'vid_name': [data_size] (str)
        # }
        # data_size should be [16275369]
        db_file = self.datapath
        db = joblib.load(db_file)

        if 'clip_images' in db and db['clip_images'][0] is None:  # No images added
            del db['clip_images']

        return db

    def _load_joints3D(self, ind, frame_ix):
        joints3D = self._joints3d[ind][frame_ix]
        return joints3D

    def _load_rotvec(self, ind, frame_ix):
        pose = self._poses[ind][frame_ix, :].reshape(-1, ROT_CONVENTION_TO_ROT_NUMBER[self.rot_convention] + 1,
                                                     3)  # +1 for global orientation
        return pose

    def _load_betas(self, ind, frame_ix):
        betas = self._betas[ind][frame_ix].transpose((1, 0))
        return betas

    def _load_gender(self, ind, frame_ix):
        gender = self._gender_to_label[self._genders[ind]]
        return gender

    def _load_body_features(self, ind, frame_ix):
        return {'mass': float(self._masses[ind]), 'height': float(self._heights[ind])}


if __name__ == "__main__":
    dataset = AMASS()
