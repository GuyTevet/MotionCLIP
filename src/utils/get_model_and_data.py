from ..datasets.get_dataset import get_datasets
from ..models.get_model import get_model as get_gen_model
import clip


def get_model_and_data(parameters, split="train"):

    # clip_model, preprocess = clip.load("ViT-B/32", device=device)  # Must set jit=False for training
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=parameters['device'], jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    datasets = get_datasets(parameters, clip_preprocess, split)
    model = get_gen_model(parameters, clip_model)
    return model, datasets
