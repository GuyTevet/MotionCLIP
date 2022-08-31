import os
from .base import argparse, adding_cuda, load_args

    
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpointname")

    group = parser.add_argument_group('Finetunning options (what should change)')
    group.add_argument("--num_epochs", type=int, help="new number of epochs of training")
    group.add_argument("--batch_size", type=int, help="size of the batches")
    group.add_argument("--lr", type=float, help="AdamW: learning rate")
    group.add_argument("--snapshot", type=int, help="frequency of saving model/viz")
    group.add_argument("--num_frames", default=-2, type=int, help="number of frames or -1 => whole, -2 => random between min_len and total")
    group.add_argument("--min_len", default=60, type=int, help="number of frames minimum per sequence or -1")
    group.add_argument("--max_len", default=100, type=int, help="number of frames maximum per sequence or -1")
    
    opt = parser.parse_args()
    
    folder, checkpoint = os.path.split(opt.checkpointname)
    parameters = load_args(os.path.join(folder, "opt.yaml"))
    parameters["folder"] = folder
    
    adding_cuda(parameters)
    epoch = int(checkpoint.split("_")[-1].split('.')[0])
    return parameters, folder, checkpoint, epoch
