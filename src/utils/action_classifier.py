import sys
sys.path.append('.')
import os
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.utils.misc import load_model_wo_clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.tensors import collate
import clip
from src.visualize.visualize import get_gpu_device
from src.utils.action_label_to_idx import action_label_to_idx

if __name__ == '__main__':
    parameters, folder, checkpointname, epoch = parser(checkpoint=True)
    gpu_device = get_gpu_device()
    parameters["device"] = f"cuda:{gpu_device}"
    data_split = 'vald'  # Hardcoded
    parameters['use_action_cat_as_text_labels'] = True
    parameters['only_60_classes'] = True

    TOP_K_METRIC = 5

    model, datasets = get_model_and_data(parameters, split=data_split)
    dataset = datasets["train"]

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)
    model.eval()

    iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                          shuffle=False, num_workers=8, collate_fn=collate)

    action_text_labels = list(action_label_to_idx.keys())
    action_text_labels.sort(key=lambda x: action_label_to_idx[x])

    texts = clip.tokenize(action_text_labels[:60]).to(model.device)
    classes_text_emb = model.clip_model.encode_text(texts).float()

    correct_preds_top_5, correct_preds_top_1 = 0,0
    total_samples = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            if isinstance(batch['x'], list):
                continue
            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(parameters['device'])
            batch = model(batch)
            texts = clip.tokenize(batch['clip_text']).to(model.device)
            batch['clip_text_embed'] = model.clip_model.encode_text(texts).float()
            labels = list(map(lambda x: [action_label_to_idx[cat] for cat in x], batch['all_categories']))
            classes_text_emb_norm = classes_text_emb / classes_text_emb.norm(dim=-1, keepdim=True)
            motion_features_norm = batch['z'] / batch['z'].norm(dim=-1, keepdim=True)
            scores = motion_features_norm @ classes_text_emb_norm.t()
            similarity = (100.0 * motion_features_norm @ classes_text_emb_norm.t()).softmax(dim=-1)

            total_samples += similarity.shape[0]
            for i in range(similarity.shape[0]):
                values, indices = similarity[i].topk(5)

                # TOP-5 CHECK
                if any([gt_cat_idx in indices for gt_cat_idx in labels[i]]):
                    correct_preds_top_5 += 1

                # TOP-1 CHECK
                values = values[:1]
                indices = indices[:1]
                if any([gt_cat_idx in indices for gt_cat_idx in labels[i]]):
                    correct_preds_top_1 += 1

            # print(f"Current Top-5 Acc. : {100 * correct_preds_top_5 / total_samples:.2f}%")

        print(f"Top-5 Acc. : {100 * correct_preds_top_5 / total_samples:.2f}%  ({correct_preds_top_5}/{total_samples})")
        print(f"Top-1 Acc. : {100 * correct_preds_top_1 / total_samples:.2f}%  ({correct_preds_top_1}/{total_samples})")
