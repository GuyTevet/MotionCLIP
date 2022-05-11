from src.models.get_model import LOSSES, JOINTSTYPES


def add_model_options(parser):
    group = parser.add_argument_group('Model options')
    group.add_argument("--modelname", default='motionclip_transformer_rc_rcxyz_vel', help="Choice of the model, should be like motionclip_transformer_rc_rcxyz_vel")
    group.add_argument("--latent_dim", default=256, type=int, help="dimensionality of the latent space")
    group.add_argument("--lambda_rc", default=1.0, type=float, help="weight of the rc divergence loss")
    group.add_argument("--lambda_rcxyz", default=1.0, type=float, help="weight of the rc divergence loss")
    group.add_argument("--lambda_vel", default=1.0, type=float, help="weight of the vel divergence loss")
    group.add_argument("--lambda_velxyz", default=1.0, type=float, help="weight of the vel divergence loss")

    group.add_argument("--jointstype", default="vertices", choices=JOINTSTYPES, help="Jointstype for training with xyz")

    group.add_argument('--vertstrans', dest='vertstrans', action='store_true', help="Training with vertex translations in the SMPL mesh")
    group.add_argument('--no-vertstrans', dest='vertstrans', action='store_false', help="Training without vertex translations in the SMPL mesh")
    group.set_defaults(vertstrans=False)

    group.add_argument("--num_layers", default=8, type=int, help="Number of layers for GRU and transformer")
    group.add_argument("--activation", default="gelu", help="Activation for function for the transformer layers")

    # Ablations
    group.add_argument("--ablation", choices=[None, "average_encoder", "zandtime", "time_encoding", "concat_bias", "extra_token"],
                       help="Ablations for the transformer architechture")

    # CLIP related losses
    group.add_argument("--clip_image_losses", default='', help="supports multiple, underscore separated, valid options are [mse, ce]. if empty, will not train on images.")
    group.add_argument("--clip_text_losses", default='', help="supports multiple, underscore separated, valid options are [mse, ce]. if empty, will not train on text.")
    group.add_argument("--clip_lambda_mse", default=1.0, type=float, help="weight of the MSE loss, for both texts and images, if in use.")
    group.add_argument("--clip_lambda_ce", default=1.0, type=float, help="weight of the CROSS-ENTROPY loss, for both texts and images, if in use.")
    group.add_argument("--clip_lambda_cosine", default=1.0, type=float, help="weight of the Cosine-dist loss, for both texts and images, if in use.")

    group.add_argument("--normalize_encoder_output", action='store_true', help="Choose if normalize the outputs of the encoder during forward")

    group.add_argument("--normalize_decoder_input", action='store_true',
                       help="Choose if normalize the input of the encoder during forward")
    group.add_argument("--use_generation_losses", action='store_true', help="Generate during training")



def parse_modelname(modelname):
    modeltype, archiname, *losses = modelname.split("_")

    if len(losses) == 0:
        raise NotImplementedError("You have to specify at least one loss function.")

    for loss in losses:
        if loss not in LOSSES:
            raise NotImplementedError("This loss is not implemented.")

    return modeltype, archiname, losses
