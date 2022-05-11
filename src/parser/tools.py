import os
import yaml


def save_args(opt, folder):
    os.makedirs(folder, exist_ok=True)
    
    # Save as yaml
    optpath = os.path.join(folder, "opt.yaml")
    with open(optpath, 'w') as opt_file:
        yaml.dump(opt, opt_file)


def load_args(filename):
    with open(filename, "rb") as optfile:
        opt = yaml.load(optfile, Loader=yaml.Loader)
    return opt


