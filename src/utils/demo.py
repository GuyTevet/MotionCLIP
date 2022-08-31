import sys

sys.path.append('')
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.utils.misc import load_model_wo_clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.tensors import collate
import clip
from src.models.get_model import get_model

cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

if __name__ == '__main__':
    parameters, folder, checkpointname, epoch = parser()

    data_split = 'vald'  # Hardcoded
    parameters[
        'datapath'] = '/disk2/briangordon/NEW_MOTION_CLIP/MotionClip/ACTOR/PUBLIC_AMASS_DIR/amass_30fps_legacy_clip_images_v02_db.pt'  # FIXME - hardcoded

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=parameters['device'],
                                            jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    # model = get_model(parameters, clip_model)
    model, datasets = get_model_and_data(parameters, split=data_split)
    dataset = datasets["train"]

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    target_path = os.path.join(os.path.dirname(checkpointpath), f'motion_rep_{data_split}_{epoch}.npz')
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    # model.load_state_dict(state_dict)
    load_model_wo_clip(model, state_dict)

    iterator = DataLoader(dataset, batch_size=2,  # parameters["batch_size"],
                          shuffle=False, num_workers=8, collate_fn=collate)

    keep_keys = ['x', 'z', 'clip_text', 'clip_path']
    buf = {}

    filename = 'generated_014_fixed'
    modi_motion = np.load(f'/disk2/briangordon/good_examples/{filename}.npy', allow_pickle=True)
    modi_motion = modi_motion.astype('float32')
    modi_motion = torch.from_numpy(modi_motion).to(model.device)
    lenghts = torch.zeros(1, device=model.device) + 64
    mask = torch.ones((1, 64), dtype=torch.bool, device=model.device)
    y = torch.zeros(1, dtype=torch.long, device=model.device)

    batch = {
        'x': modi_motion,
        'mask': mask,
        'lenghts': lenghts,
        'y': y
    }
    batch.update(model.encoder(batch))
    batch["z"] = batch["mu"]
    # Encode text with clip
    texts = clip.tokenize(['walk', 'run', 'lie', 'sit', 'swim']).to(parameters['device'])
    text_features = clip_model.encode_text(texts).float()

    # normalized features motion & text
    features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    seq_motion_features_norm = batch["z"] / batch["z"].norm(dim=-1, keepdim=True)
    cos = cosine_sim(features_norm, seq_motion_features_norm)

    print(cos)
    input('look')
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            # print('batch', {k: type(v) for k,v in batch.items()})
            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(parameters['device'])
            # batch = {key: val.to(parameters['device']) for key, val in batch.items()}
            # print('batch', {k: v.shape for k,v in batch.items()})
            print(f'x: {batch["x"].shape}')
            print(f'mask: {batch["mask"].shape}')
            print(f'lengths: {batch["lengths"].shape}')
            print(f'y: {batch["y"].shape}')

            # print(f'x: {batch["x"][0, :, :, 1]}')

            # print('batch', {k: v for k,v in batch.items()})

            input('look')
            if model.outputxyz:
                batch["x_xyz"] = model.rot2xyz(batch["x"], batch["mask"])
            elif model.pose_rep == "xyz":
                batch["x_xyz"] = batch["x"]

            # Encode Motion - Encoded motion will be under "z" key.
            batch.update(model.encoder(batch))
            batch["z"] = batch["mu"]

            # Encode text with clip
            texts = clip.tokenize(batch['clip_text']).to(parameters['device'])
            text_features = clip_model.encode_text(texts).float()

            # normalized features motion & text
            features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            seq_motion_features_norm = batch["z"] / batch["z"].norm(dim=-1, keepdim=True)
            cos = cosine_sim(features_norm, seq_motion_features_norm)
            cosine_loss = (1 - cos).mean()

            # batch = model(batch)
            # print('batch', {k: v.shape for k,v in batch.items()})

            """ LOGIC TO SAVE OUTPUTS TO NPZ FILE - not required 100%"""
            if len(buf) == 0:
                for k in keep_keys:
                    _to_write = batch[k].cpu().numpy() if torch.is_tensor(batch[k]) else np.array(batch[k])
                    buf[k] = _to_write
            else:
                for k in keep_keys:
                    _to_write = batch[k].cpu().numpy() if torch.is_tensor(batch[k]) else np.array(batch[k])
                    buf[k] = np.concatenate((buf[k], _to_write), axis=0)
            print('buf', {k: v.shape for k, v in buf.items()})
            # print('clip_text', buf['clip_text'])
            # print('clip_path', buf['clip_path'])

            # FIXME - for now we need just a sample - hence, adding an early stop
            if i == 5:
                break

    print(f'Saving {target_path}')
    np.savez(target_path, buf)
