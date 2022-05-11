# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


import joblib
import argparse
from tqdm import tqdm
import json
import os.path as osp
import os
import sys
sys.path.append('.')

import torch
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from src.datasets import smpl_utils
from src import config
import numpy as np
from PIL import Image

comp_device = torch.device("cpu")

dict_keys = ['betas', 'dmpls', 'gender', 'mocap_framerate', 'poses', 'trans']
action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]  # [18,]

def get_joints_to_use(args):
    joints_to_use = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 37
    ])  # 23 joints + global_orient # 21 base joints + left_index1(22) + right_index1 (37)
    return np.arange(0, len(smpl_utils.SMPLH_JOINT_NAMES) * 3).reshape((-1, 3))[joints_to_use].reshape(-1)

framerate_hist = []

all_sequences = [
    'ACCAD',
    'BioMotionLab_NTroje',
    'CMU',
    'EKUT',
    'Eyes_Japan_Dataset',
    'HumanEva',
    'KIT',
    'MPI_HDM05',
    'MPI_Limits',
    'MPI_mosh',
    'SFU',
    'SSM_synced',
    'TCD_handMocap',
    'TotalCapture',
    'Transitions_mocap',
]
amass_test_split = ['Transitions_mocap', 'SSM_synced']
amass_vald_split = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh']
amass_train_split = ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje', 'EKUT', 'TCD_handMocap', 'ACCAD']
# Source - https://github.com/nghorbani/amass/blob/08ca36ce9b37969f72d7251eb61564a7fd421e15/src/amass/data/prepare_data.py#L235
amass_splits = {
    'test': amass_test_split,
    'vald': amass_vald_split,
    'train': amass_train_split
}
amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))  # no change - just making sure
assert len(amass_splits['train'] + amass_splits['test'] + amass_splits['vald']) == len(all_sequences) == 15

def read_data(folder, split_name,dataset_name, target_fps, max_fps_dist, joints_to_use, quick_run,babel_labels, clip_images_dir=None):
    # sequences = [osp.join(folder, x) for x in sorted(os.listdir(folder)) if osp.isdir(osp.join(folder, x))]
    # target_fps = None -> will not resample (for backward compatibility)

    if dataset_name == "amass":
        sequences = amass_splits[split_name]
    else:
        sequences = all_sequences

    db = {
        'vid_names': [],
        'thetas': [],
        'joints3d': [],
        'clip_images': [],
        'clip_pathes': [],
        'text_raw_labels': [],
        'text_proc_labels': [],
        'action_cat': []
    }

    # instance SMPL model
    print('Loading Body Models')
    body_models = {
        'neutral': BodyModel(config.SMPLH_AMASS_MODEL_PATH, num_betas=config.NUM_BETAS).to(comp_device),
    }
    print('DONE! - Loading Body Models')

    clip_images_path = clip_images_dir
    assert os.path.isdir(clip_images_path)


    for seq_name in sequences:
        print(f'Reading {seq_name} sequence...')
        seq_folder = osp.join(folder, seq_name)

        results_dict = read_single_sequence(split_name, dataset_name, seq_folder, seq_name, body_models, target_fps,
                                            max_fps_dist, joints_to_use, quick_run, clip_images_path, babel_labels)

        for k in db.keys(): db[k].extend(results_dict[k])

    return db


def read_single_sequence(split_name, dataset_name, folder, seq_name, body_models, target_fps, max_fps_dist,
                         joints_to_use, quick_run, clip_images_path, fname_to_babel):
    # target_fps = None -> will not resample (for backward compatibility)
    subjects = os.listdir(folder)

    thetas = []
    vid_names = []
    joints3d = []
    clip_images = []
    clip_pathes = []
    text_raw_labels = []
    text_proc_labels = []
    action_cat = []

    for subject in tqdm(subjects):
        actions = [x for x in os.listdir(osp.join(folder, subject)) if x.endswith('.npz')]

        for action in actions:
            fname = osp.join(folder, subject, action)
            if fname.endswith('shape.npz'):
                continue

            # Remove folder path from fname
            folder_path, sequence_name = os.path.split(folder)
            seq_subj_action = osp.join(sequence_name, subject, action)
            if seq_subj_action in fname_to_babel:
                babel_dict = fname_to_babel[seq_subj_action]
            else:
                print(f"Not in BABEL: {seq_subj_action}")
                continue

            if dataset_name == "babel":
                # # Check if pose belongs to split
                babel_split = babel_dict['split'].replace("val", "vald")  # Fix diff in split name
                if babel_split != split_name:
                    continue

            data = np.load(fname)
            duration_t = babel_dict['dur']
            fps = data['poses'].shape[0] / duration_t

            # Seq. labels
            seq_raw_labels, seq_proc_label, seq_act_cat = [], [], []
            frame_raw_text_labels = np.full(data['poses'].shape[0], "", dtype=np.object)
            frame_proc_text_labels = np.full(data['poses'].shape[0], "", dtype=np.object)
            frame_action_cat = np.full(data['poses'].shape[0], "", dtype=np.object)

            for label_dict in babel_dict['seq_ann']['labels']:
                seq_raw_labels.extend([label_dict['raw_label']])
                seq_proc_label.extend([label_dict['proc_label']])
                if label_dict['act_cat'] is not None:
                    seq_act_cat.extend(label_dict['act_cat'])

            # Frames labels
            if babel_dict['frame_ann'] is None:
                frame_raw_labels = "and ".join(seq_raw_labels)
                frame_proc_labels = "and ".join(seq_proc_label)
                start_frame = 0
                end_frame = data['poses'].shape[0]
                frame_raw_text_labels[start_frame:end_frame] = frame_raw_labels
                frame_proc_text_labels[start_frame:end_frame] = frame_proc_labels
                frame_action_cat[start_frame:end_frame] = ",".join(seq_act_cat)
            else:
                for label_dict in babel_dict['frame_ann']['labels']:
                    start_frame = round(label_dict['start_t'] * fps)
                    end_frame = round(label_dict['end_t'] * fps)
                    frame_raw_text_labels[start_frame:end_frame] = label_dict['raw_label']
                    frame_proc_text_labels[start_frame:end_frame] = label_dict['proc_label']
                    if label_dict['act_cat'] is not None:
                        frame_action_cat[start_frame:end_frame] = str(",".join(label_dict['act_cat']))

            if target_fps is not None:
                mocap_framerate = float(data['mocap_framerate'])
                sampling_freq = round(mocap_framerate / target_fps)
                if abs(mocap_framerate / float(sampling_freq) - target_fps) > max_fps_dist:
                    print('Will not sample [{}]fps seq with sampling_freq [{}], since target_fps=[{}], max_fps_dist=[{}]'
                          .format(mocap_framerate, sampling_freq, target_fps, max_fps_dist))
                    continue
                # pose = data['poses'][:, joints_to_use]
                pose = data['poses'][0::sampling_freq, joints_to_use]
                pose_all = data['poses'][0::sampling_freq, :]
                frame_raw_text_labels = frame_raw_text_labels[0::sampling_freq]
                frame_proc_text_labels = frame_proc_text_labels[0::sampling_freq]

            else:
                # don't sample
                pose = data['poses'][:, joints_to_use]
                pose_all = data['poses'][:, :]

            if pose.shape[0] < 60:
                continue

            theta = pose
            vid_name = np.array([f'{seq_name}_{subject}_{action[:-4]}']*pose.shape[0])

            if quick_run:
                joints = None
                images = None
            else:
                root_orient = torch.Tensor(pose_all[:, :(smpl_utils.JOINTS_SART_INDEX * 3)]).to(comp_device)
                pose_hand = torch.Tensor(pose_all[:, (smpl_utils.L_HAND_START_INDEX * 3):]).to(comp_device)
                pose_body = torch.Tensor(pose_all[:, (smpl_utils.JOINTS_SART_INDEX * 3):(
                            smpl_utils.L_HAND_START_INDEX * 3)]).to(comp_device)
                body_model = body_models['neutral']

                body_motion = body_model(pose_body=pose_body, pose_hand=pose_hand, root_orient=root_orient)
                joints = c2c(body_motion.Jtr)  # [seq_len, 52, 3]
                joints = joints[:, action2motion_joints]  # [seq_len, 18, 3]

                images = None
                images_path = None
                if clip_images_path is not None:
                    images_path = [os.path.join(clip_images_path, f) for f in os.listdir(clip_images_path) if f.startswith(vid_name[0]) and f.endswith('.png')]
                    images_path.sort(key=lambda x: int(x.replace('.png', '').split('frame')[-1]))
                    images_path = np.array(images_path)
                    images = [np.asarray(Image.open(im)) for im in images_path]
                    images = np.concatenate([np.expand_dims(im, 0) for im in images], axis=0)

            vid_names.append(vid_name)
            thetas.append(theta)
            joints3d.append(joints)
            clip_images.append(images)
            clip_pathes.append(images_path)
            text_raw_labels.append(frame_raw_text_labels)
            text_proc_labels.append(frame_proc_text_labels)
            action_cat.append(frame_action_cat)


    # return np.concatenate(thetas, axis=0), np.concatenate(vid_names, axis=0)
    return {
        # 'betas': betas,
        'vid_names': vid_names,
        'thetas': thetas,
        'joints3d': joints3d,
        'clip_images': clip_images,
        'clip_pathes': clip_pathes,
        'text_raw_labels': text_raw_labels,
        'text_proc_labels': text_proc_labels,
        'action_cat': action_cat
    }


def get_babel_labels(babel_dir_path):
    print("Loading babel labels")
    l_babel_dense_files = ['train', 'val']
    # BABEL Dataset
    pose_file_to_babel = {}
    for file in l_babel_dense_files:
        path = os.path.join(babel_dir_path, file + '.json')
        data = json.load(open(path))
        for seq_id, seq_dict in data.items():
            npz_path = os.path.join(*(seq_dict['feat_p'].split(os.path.sep)[1:]))
            seq_dict['split'] = file
            pose_file_to_babel[npz_path] = seq_dict
    print("DONE! - Loading babel labels")
    return pose_file_to_babel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='dataset directory', default='./data/amass')
    parser.add_argument('--output_dir', type=str, help='target directory', default='./data/amass_db')
    parser.add_argument('--clip_images_dir', type=str, help='dataset directory', default='./data/render')
    parser.add_argument('--target_fps', type=int, choices=[10, 30, 60], default=30)
    parser.add_argument('--quick_run', action='store_true', help='quick_run wo saving and modeling 3d positions with smpl, just for debug')
    parser.add_argument('--dataset_name',required=True, type=str, choices=['amass', 'babel'],
                        help='choose which dataset you want to create')
    parser.add_argument('--babel_dir', type=str, help='path to processed BABEL downloaded dir BABEL file',
                        default='./data/babel_v1.0_release')

    args = parser.parse_args()

    fname_to_babel = get_babel_labels(args.babel_dir)

    joints_to_use = get_joints_to_use(args)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    max_fps_dist = 5 # max distance from target fps that can be tolerated
    if args.quick_run:
        print('quick_run mode')

    for split_name in amass_splits.keys():
        db = read_data(args.input_dir,
                       split_name=split_name,
                       dataset_name=args.dataset_name,
                       target_fps=args.target_fps,
                       max_fps_dist=max_fps_dist,
                       joints_to_use=joints_to_use,
                       quick_run=args.quick_run,
                       babel_labels=fname_to_babel,
                       clip_images_dir=args.clip_images_dir
                       )


        db_file = osp.join(args.output_dir, '{}_{}fps'.format(args.dataset_name, args.target_fps))
        db_file += '_{}.pt'.format(split_name)
        if args.quick_run:
            print(f'quick_run mode - file should be saved to {db_file}')
        else:
            print(f'Saving AMASS dataset to {db_file}')
            joblib.dump(db, db_file)


