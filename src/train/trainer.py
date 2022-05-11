import torch
from torch import nn
from tqdm import tqdm
# import CLIP.clip as clip
from PIL import Image
from copy import deepcopy

# THIS CODE WAS MIGRATED TO motionclip.py
# loss_seq_motion = nn.CrossEntropyLoss()
# loss_txt = nn.CrossEntropyLoss()
# loss_mse = nn.MSELoss()

def train_or_test(model, optimizer, iterator, device, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    # loss of the epoch
    dict_loss = {}
    # dict_loss = {loss: 0. for loss in model.losses}
    # THIS CODE WAS MIGRATED TO motionclip.py
    # dict_loss.update({loss: 0 for loss in ['seq_motion_loss', 'txt_loss', 'clip_mixed_loss', 'mse_clip_loss','total_loss']})

    with grad_env():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            # batch_size = batch['x'].shape[0]

            # print('INPUT')
            # for key, val in batch.items():
            #     print(f"{key}: {val.shape}")
            # mixed_loss = 0
            # losses = {}

            # Put everything in device
            # Added if is_tensor as 'clip_text' in batch is a list of strings, not a tensor!
            # if len(batch['clip_text']) == 0:
            #     continue
            # print(f"Batch size: {len(batch['clip_text'])}")
            if len(batch['clip_text']) < 2:
                continue
            batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}

            if mode == "train":
                # update the gradients to zero
                optimizer.zero_grad()

            # forward pass
            batch = model(batch)

            mixed_loss, losses = model.compute_loss(batch)

            # THIS CODE WAS MIGRATED TO motionclip.py
            # if 'clip_images' in batch.keys():
            #
            #     with torch.no_grad():
            #         image_features = model.clip_model.encode_image(batch['clip_images']).float() # preprocess is done in dataloader
            #         features = image_features
            #
            # else:
            #     # Get action names from batch
            #     y_action_names = [iterator.dataset.action_to_action_name(x) for x in
            #                       batch["y"].cpu().detach().numpy().tolist()]
            #     y_action_names = [action_name.replace("-", " ").replace("_", " ") for action_name in y_action_names]
            #
            #     # Tokenize action names
            #     with torch.no_grad():
            #         texts = clip.tokenize(y_action_names).to(device)
            #         text_features = model.clip_model.encode_text(texts).float()
            #         features = text_features
            #
            # # normalized features
            # features_norm = features / features.norm(dim=-1, keepdim=True)
            # seq_motion_features_norm = batch["z"] / batch["z"].norm(dim=-1, keepdim=True)
            #
            # # print(f"text_features: {text_features.shape} {text_features.type()}")
            # # print(f"seq_motion_features: {seq_motion_features.shape} {seq_motion_features.type()}")
            # # input('look')
            # # for k in batch.keys():
            # #     print(k, batch[k].shape, batch[k].dtype)
            # logit_scale = model.clip_model.logit_scale.exp()
            # logits_per_motion = logit_scale * seq_motion_features_norm @ features_norm.t()
            # logits_per_text = logits_per_motion.t()
            #
            # # ground_truth = torch.arange(len(y_action_names), dtype=torch.long, device=device)
            # ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
            # seq_motion_loss = loss_seq_motion(logits_per_motion, ground_truth)
            # txt_loss = loss_txt(logits_per_text, ground_truth)
            # clip_mixed_loss = (seq_motion_loss + txt_loss) / 2
            # mse_clip_loss = loss_mse(features, batch["z"])
            # clip_losses = {
            #     'seq_motion_loss': seq_motion_loss.item(),
            #     'txt_loss': txt_loss.item(),
            #     'clip_mixed_loss': clip_mixed_loss.item(),
            #     'mse_clip_loss': mse_clip_loss.item()
            # }
            # losses.update(clip_losses)
            # mixed_loss += clip_mixed_loss
            # mixed_loss += mse_clip_loss
            # losses.update({'total_loss': mixed_loss.item()})

            if i == 0:
                dict_loss = deepcopy(losses)
            else:
                for key in dict_loss.keys():
                    dict_loss[key] += losses[key]

            if mode == "train":
                # backward pass
                mixed_loss.backward()
                # update the weights
                optimizer.step()

            # print('OUTPUT')
            # for key, val in batch.items():
            #     print(f"{key}: {val.shape}")
            # input('look2')

    return dict_loss


def train(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="train")


def test(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="test")
