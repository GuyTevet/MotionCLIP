import os

SMPL_DATA_PATH = "./models/smpl"
SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')

SMPLH_AMASS_PATH = './models/smplh'
SMPLH_AMASS_MODEL_PATH = os.path.join(SMPLH_AMASS_PATH, "neutral/model.npz")
SMPLH_AMASS_MALE_MODEL_PATH = os.path.join(SMPLH_AMASS_PATH, "male/model.npz")
SMPLH_AMASS_FEMALE_MODEL_PATH = os.path.join(SMPLH_AMASS_PATH, "female/model.npz")

ROT_CONVENTION_TO_ROT_NUMBER = {
    'legacy': 23,
    'no_hands': 21,
    'full_hands': 51,
    'mitten_hands': 33,
}

GENDERS = ['neutral', 'male', 'female']
NUM_BETAS = 10