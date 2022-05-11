import os

from src.models.get_model import JOINTSTYPES
from .base import ArgumentParser, add_cuda_options, adding_cuda
from .tools import load_args
from .dataset import add_dataset_options


def construct_figname(parameters):
    figname = "fig_{:03d}"
    return figname


def add_visualize_options(parser):
    group = parser.add_argument_group('Visualization options')
    group.add_argument("--num_actions_to_sample", default=5, type=int, help="num actions to sample")
    group.add_argument("--num_samples_per_action", default=5, type=int, help="num samples per action")
    group.add_argument("--fps", default=20, type=int, help="FPS for the rendering")
    group.add_argument("--appearance_mode", default='motionclip', choices=['motionclip', 'original'], type=str,
                       help="How the stick figures will appear")

    group.add_argument("--force_visu_joints", dest='force_visu_joints', action='store_true',
                       help="if we want to visualize joints even if it is rotation")
    group.add_argument('--no-force_visu_joints', dest='force_visu_joints', action='store_false',
                       help="if we don't want to visualize joints even if it is rotation")
    group.set_defaults(force_visu_joints=True)

    group.add_argument("--jointstype", default="smpl", choices=JOINTSTYPES,
                       help="Jointstype for training with xyz")
    group.add_argument('--vertstrans', dest='vertstrans', action='store_true', help="Training with vertex translations")
    group.add_argument('--no-vertstrans', dest='vertstrans', action='store_false',
                       help="Training without vertex translations")
    group.set_defaults(vertstrans=False)

    group.add_argument("--noise_same_action", default="random",
                       choices=["interpolate", "random", "same"],
                       help="inside one action, sample several noise or interpolate it")

    group.add_argument("--noise_diff_action", default="random",
                       choices=["random", "same"],
                       help="use the same noise or different noise for every actions")

    group.add_argument("--duration_mode", default="mean",
                       choices=["mean", "interpolate"],
                       help="use the same noise or different noise for every actions")

    group.add_argument("--reconstruction_mode", default="ntf",
                       choices=["tf", "ntf", "both"],
                       help="reconstruction: teacher forcing or not or both")

    group.add_argument("--decoder_test", default="new",
                       choices=["new", "diffaction", "diffduration", "interpolate_action"],
                       help="what is the test we want to do")

    group.add_argument("--fact_latent", type=int, default=1,
                       help="factor for max latent space")

    group.add_argument("--images_dir", type=str, default='./action_images',
                       help="dir with images for clip visualization")

    group.add_argument("--input_file", default=None, help="Input txt/csv file defining generation. For more info, see README.")

    group.add_argument("--zero_global_orient", action='store_true', help="will set global orientation to zero")
    group.add_argument("--ae_after_generation", action='store_true', help="Apply auto encoding after generation to project motion into motion manifold.")


def parser(checkpoint=True):
    parser = ArgumentParser()
    if checkpoint:
        parser.add_argument("checkpointname")
    else:
        add_dataset_options(parser)

    # add visualize options back
    add_visualize_options(parser)

    # cuda options
    add_cuda_options(parser)

    opt = parser.parse_args()
    if checkpoint:
        newparameters = {key: val for key, val in vars(opt).items() if val is not None}
        folder, checkpoint = os.path.split(newparameters["checkpointname"])
        parameters = load_args(os.path.join(folder, "opt.yaml"))
        parameters.update(newparameters)
    else:
        parameters = {key: val for key, val in vars(opt).items() if val is not None}

    adding_cuda(parameters)

    if checkpoint:
        parameters["figname"] = construct_figname(parameters)
        epoch = int(checkpoint.split("_")[-1].split('.')[0])
        return parameters, folder, checkpoint, epoch
    else:
        return parameters
