import os

from src.models.get_model import JOINTSTYPES
from .base import ArgumentParser, add_cuda_options, adding_cuda
from .tools import load_args


def add_generation_options(parser):
    group = parser.add_argument_group('Generation options')
    group.add_argument("--num_samples_per_action", default=5, type=int, help="num samples per action")
    group.add_argument("--num_frames", default=60, type=int, help="The number of frames considered (overrided if duration mode is chosen)")
    group.add_argument("--fact_latent", default=1, type=int, help="Fact latent")

    group.add_argument("--jointstype", default="smpl", choices=JOINTSTYPES,
                       help="Jointstype for training with xyz")

    group.add_argument('--vertstrans', dest='vertstrans', action='store_true', help="Add the vertex translations")
    group.add_argument('--no-vertstrans', dest='vertstrans', action='store_false', help="Do not add the vertex translations")
    group.set_defaults(vertstrans=False)

    group.add_argument("--mode", default="gen", choices=["interpolate", "gen", "duration", "reconstruction"],
                       help="The kind of generation considered.")


def parser():
    parser = ArgumentParser()
    parser.add_argument("checkpointname")

    # add visualize options back
    add_generation_options(parser)

    # cuda options
    add_cuda_options(parser)

    opt = parser.parse_args()
    newparameters = {key: val for key, val in vars(opt).items() if val is not None}
    folder, checkpoint = os.path.split(newparameters["checkpointname"])
    parameters = load_args(os.path.join(folder, "opt.yaml"))
    parameters.update(newparameters)

    adding_cuda(parameters)

    epoch = int(checkpoint.split("_")[-1].split('.')[0])
    return parameters, folder, checkpoint, epoch
