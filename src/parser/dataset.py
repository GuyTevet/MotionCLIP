from src.datasets.dataset import POSE_REPS


def add_dataset_options(parser):
    group = parser.add_argument_group('Dataset options')
    group.add_argument("--dataset", required=True, help="Dataset to load", default='amass')
    group.add_argument("--datapath", help="Path of the data")
    group.add_argument("--num_frames", required=True, type=int, help="number of frames or -1 => whole, -2 => random between min_len and total")
    group.add_argument("--sampling", default="conseq", choices=["conseq", "random_conseq", "random"], help="sampling choices")
    group.add_argument("--sampling_step", default=1, type=int, help="sampling step")
    group.add_argument("--pose_rep", required=True, choices=POSE_REPS, help="xyz or rotvec etc")

    group.add_argument("--max_len", default=-1, type=int, help="number of frames maximum per sequence or -1")
    group.add_argument("--min_len", default=-1, type=int, help="number of frames minimum per sequence or -1")
    group.add_argument("--num_seq_max", default=-1, type=int, help="number of sequences maximum to load or -1")

    group.add_argument("--glob", dest='glob', action='store_true', help="if we want global rotation")
    group.add_argument('--no-glob', dest='glob', action='store_false', help="if we don't want global rotation")
    group.set_defaults(glob=True)
    group.add_argument("--glob_rot", type=int, nargs="+", default=[3.141592653589793, 0, 0],
                       help="Default rotation, usefull if glob is False")
    group.add_argument("--translation", dest='translation', action='store_true',
                       help="if we want to output translation")
    group.add_argument('--no-translation', dest='translation', action='store_false',
                       help="if we don't want to output translation")
    group.set_defaults(translation=True)

    group.add_argument("--debug", dest='debug', action='store_true', help="if we are in debug mode")
    group.set_defaults(debug=False)

    group.add_argument('--use_action_cat_as_text_labels', action='store_true', help="If true, dataset on loading will align root pose(rotation) to be the unit")
    group.add_argument('--only_60_classes', action='store_true', help="If true, dataset on loading will align root pose(rotation) to be the unit")
    group.add_argument('--use_only_15_classes', action='store_true', help="If true, We use only the 15 most frequence classes from BABEL")
