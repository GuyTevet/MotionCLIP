import os
import sys
sys.path.append('.')

import csv
import matplotlib.pyplot as plt
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.visualize.visualize import viz_clip_interp, get_gpu_device
from src.utils.misc import load_model_wo_clip


import src.utils.fixseed  # noqa

plt.switch_backend('agg')


def main():
    # parse options
    parameters, folder, checkpointname, epoch = parser()
    gpu_device = get_gpu_device()
    parameters["device"] = f"cuda:{gpu_device}"
    model, datasets = get_model_and_data(parameters, split='all')
    num_stops = 5  # FIXME - hardcoded

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)

    assert os.path.isfile(parameters['input_file'])
    with open(parameters['input_file'], 'r') as fr:
        interp_csv = list(csv.DictReader(fr))
    viz_clip_interp(model, datasets, interp_csv, num_stops, epoch, parameters, folder=folder)

if __name__ == '__main__':
    main()
