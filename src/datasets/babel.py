import os
import json
from argparse import ArgumentParser  # noqa
from os.path import join as ospj


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--babel_dir")
    opt = parser.parse_args()
    parameters = {key: val for key, val in vars(opt).items() if val is not None}

    l_babel_dense_files = ['train', 'val']

    # BABEL Dataset
    pose_file_to_babel = {}
    babel = {}
    for file in l_babel_dense_files:
        path = ospj(parameters['babel_dir'], file + '.json')
        data = json.load(open(path))
        babel[file] = data
        for seq_id, seq_dict in data.items():
            npz_path = ospj(*(seq_dict['feat_p'].split(os.path.sep)[1:]))
            seq_dict['split'] = file
            pose_file_to_babel[npz_path] = seq_dict
    with open(ospj(parameters['babel_dir'], 'pose_file_to_babel_train_val.json'), 'w') as f:
        json.dump(pose_file_to_babel, f)
