import numpy as np
import torch
import torch.nn as nn

import clip
from ..tools.losses import get_loss_function
from ..rotation2xyz import Rotation2xyz

loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()

cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


class MOTIONCLIP(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, clip_lambdas={}, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.outputxyz = outputxyz

        self.lambdas = lambdas
        self.clip_lambdas = clip_lambdas

        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans

        self.clip_model = kwargs['clip_model']
        assert self.clip_model.training == False  # make sure clip is frozen

        self.align_pose_frontview = kwargs.get('align_pose_frontview', False)
        self.use_generation_losses = kwargs.get('use_generation_losses', False)

        self.losses = list(self.lambdas) + ["mixed"]

        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}

    def rot2xyz(self, x, mask, get_rotations_back=False, **kwargs):
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask, get_rotations_back=get_rotations_back, **kargs)


    def compute_loss(self, batch):

        # compute all losses other than clip
        mixed_loss = 0.
        losses = {}
        for ltype, lam in self.lambdas.items():
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            mixed_loss += loss * lam
            losses[ltype] = loss.item()

        # compute clip losses
        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)

        # mix and add clip losses
        mixed_loss_with_clip = mixed_loss + mixed_clip_loss  # this is the ultimate loss to optimize, combining ALL losses
        losses.update(clip_losses)
        losses["mixed_without_clip"] = mixed_loss.item()
        losses["mixed_with_clip"] = mixed_loss_with_clip.item()
        if not isinstance(mixed_clip_loss, float):
            losses["mixed_clip_only"] = mixed_clip_loss.item()
        else:
            losses["mixed_clip_only"] = mixed_clip_loss

        # Generating from text losses:
        if self.use_generation_losses:
            batch.update(self.decoder(batch, use_text_emb=True))
            # if we want to output xyz
            if self.outputxyz:
                batch["txt_output_xyz"] = self.rot2xyz(batch["txt_output"], batch["mask"])
            elif self.pose_rep == "xyz":
                batch["txt_output_xyz"] = batch["output"]

            gen_mixed_loss = 0.
            for ltype, lam in self.lambdas.items():
                loss_function = get_loss_function(ltype)
                loss = loss_function(self, batch, use_txt_output=True)
                gen_mixed_loss += loss * lam
                losses[f'gen_{ltype}'] = loss.item()

            mixed_loss_with_clip += gen_mixed_loss

        return mixed_loss_with_clip, losses

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}
        for d in self.clip_lambdas.keys():
            if len(self.clip_lambdas[d].keys()) == 0:
                continue
            with torch.no_grad():
                if d == 'image':
                    if 'clip_images_emb' in batch:
                        d_features = batch['clip_images_emb'].float()
                    else:
                        d_features = self.clip_model.encode_image(
                            batch['clip_images']).float()  # preprocess is done in dataloader
                elif d == 'text':
                    texts = clip.tokenize(batch['clip_text']).to(self.device)
                    d_features = self.clip_model.encode_text(texts).float()
                    batch['clip_text_emb'] = d_features
                else:
                    raise ValueError(f'Invalid clip domain [{d}]')

            motion_features = batch['z']

            # normalized features
            d_features_norm = d_features / d_features.norm(dim=-1, keepdim=True)
            motion_features_norm = motion_features / motion_features.norm(dim=-1, keepdim=True)
            if 'ce' in self.clip_lambdas[d].keys():
                logit_scale = self.clip_model.logit_scale.exp()
                logits_per_motion = logit_scale * motion_features_norm @ d_features_norm.t()
                logits_per_d = logits_per_motion.t()

                batch_size = batch['x'].shape[0]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)

                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                mixed_clip_loss += clip_mixed_loss * self.clip_lambdas[d]['ce']

            if 'mse' in self.clip_lambdas[d].keys():
                mse_clip_loss = loss_mse(d_features, motion_features)
                clip_losses[f'{d}_mse'] = mse_clip_loss.item()
                mixed_clip_loss += mse_clip_loss * self.clip_lambdas[d]['mse']

            if 'cosine' in self.clip_lambdas[d].keys():
                cos = cosine_sim(d_features_norm, motion_features_norm)
                cosine_loss = (1 - cos).mean()
                clip_losses[f'{d}_cosine'] = cosine_loss.item()
                mixed_clip_loss += cosine_loss * self.clip_lambdas[d]['cosine']

        return mixed_clip_loss, clip_losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate(self, classes, durations, nspa=1,
                 # noise_same_action="random", noise_diff_action="random",
                 # fact=1,
                 is_amass=False, is_clip_features=False,
                 # input_type="motion",
                 textual_labels=None):
        clip_dim = self.clip_model.ln_final.normalized_shape[0]
        if is_clip_features:
            # assumed dims: classes [nspa, nats, 512]
            assert len(classes.shape) == 3
            assert classes.shape[-1] == clip_dim
            clip_features = classes.reshape([-1, clip_dim])
            nspa, nats = classes.shape[:2]
            # y = torch.zeros(y_action_names.shape, dtype=int)

            # raise NotImplementedError('You trained with mappers, but the inverse of mapping is not yet implemented! so inference is not possible!')

            y = clip_features
            if textual_labels is not None:
                y = np.array(textual_labels).reshape([-1])

        if len(durations.shape) == 1:
            lengths = durations.to(self.device).repeat(nspa)
        else:
            lengths = durations.to(self.device).reshape(clip_features.shape[0])

        mask = self.lengths_to_mask(lengths)

        batch = {"z": clip_features,  # fact*z,
                 "y": y,
                 "mask": mask, "lengths": lengths}

        if not is_clip_features:
            batch['y'] = y

        batch = self.decoder(batch)

        if is_amass and not self.align_pose_frontview:  # lose global orientation for amass dataset
            print('NOTE: eliminating global orientation')
            batch['output'][:, 0] = torch.tensor([1, 0, 0, 0, 1, 0]).unsqueeze(0).unsqueeze(2)

        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]

        return batch

    def forward(self, batch):
        if self.outputxyz:
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))
        batch["z"] = batch["mu"]
        # decode
        batch.update(self.decoder(batch))

        # if we want to output xyz
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch
