# Source: https://github.com/vchoutas/smplx/blob/master/smplx/joint_names.py
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
import numpy as np


# SMPLX
JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]

SMPLH_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]

# Added manually according to https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
SMPL_JOINT_NAMES = [
    'Pelvis',  # 0
    'L_Hip',  # 1
    'R_Hip',  # 2
    'Spine1',  # 3
    'L_Knee',  # 4
    'R_Knee',  # 5
    'Spine2',  # 6
    'L_Ankle',  # 7
    'R_Ankle',  # 8
    'Spine3',  # 9
    'L_Foot',  # 10
    'R_Foot',  # 11
    'Neck',  # 12
    'L_Collar',  # 13
    'R_Collar',  # 14
    'Head',  # 15
    'L_Shoulder',  # 16
    'R_Shoulder',  # 17
    'L_Elbow',  # 18
    'R_Elbow',  # 19
    'L_Wrist',  # 20
    'R_Wrist',  # 21
    'L_Hand',  # 22
    'R_Hand',  # 23
]


SMPLH_NUM_JOINTS = 52
SMPLH_JOINT_NAMES = SMPLH_JOINT_NAMES[:SMPLH_NUM_JOINTS]

# joint sub-sets (on top of SMPLH_JOINT_NAMES)
HAND_FULL_JOINTS = np.arange(15)
HAND_2_JOINTS = np.concatenate([np.arange(3), np.arange(12, 15)])
R_HAND_START_INDEX = 37
L_HAND_START_INDEX = 22
JOINTS_SART_INDEX = 1
BASE_JOINTS = np.arange(22)
R_HAND_FULL = HAND_FULL_JOINTS + R_HAND_START_INDEX
R_HAND_2 = HAND_2_JOINTS + R_HAND_START_INDEX
L_HAND_FULL = HAND_FULL_JOINTS + L_HAND_START_INDEX
L_HAND_2 = HAND_2_JOINTS + L_HAND_START_INDEX


# For analysis
ROOT_STATS = {name: SMPLH_JOINT_NAMES.index(name) for name in ['pelvis']}
# R_HAND_STATS = {name: SMPLH_JOINT_NAMES.index(name) for name in [
#     'right_index1', 'right_index2', 'right_index3', 'right_thumb1', 'right_thumb2', 'right_thumb3']}
R_HAND_STATS = {name: SMPLH_JOINT_NAMES.index(name) for name in [
    'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_index1', 'right_index2', 'right_index3', 'right_thumb1', 'right_thumb2', 'right_thumb3']}
L_HAND_STATS = {name: SMPLH_JOINT_NAMES.index(name) for name in [
    'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_index1', 'left_index2', 'left_index3', 'left_thumb1', 'left_thumb2', 'left_thumb3']}
R_LEG_STATS = {name: SMPLH_JOINT_NAMES.index(name) for name in [
    'right_hip', 'right_knee', 'right_ankle']} # 'right_foot' is static
R_ARM_STATS = {name: SMPLH_JOINT_NAMES.index(name) for name in [
    'right_shoulder', 'right_elbow', 'right_wrist']}
