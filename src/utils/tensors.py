import torch
import numpy as np

def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    if len(notnone_batches) == 0:
        out_batch = {"x": [], "y": [],
                     "mask": [], "lengths": [],
                     "clip_image": [], "clip_text": [],
                     "clip_path": [], "clip_images_emb": []
                     }
        return out_batch
    databatch = [b['inp'] for b in notnone_batches]
    labelbatch = [b['target'] for b in notnone_batches]
    lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor)


    out_batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
             # "y_action_names": actionlabelbatchTensor}
    if 'clip_image' in notnone_batches[0]:
        clip_image_batch = [torch.as_tensor(b['clip_image']) for b in notnone_batches]
        out_batch.update({'clip_images': collate_tensors(clip_image_batch)})

    if 'clip_text' in notnone_batches[0]:
        textbatch = [b['clip_text'] for b in notnone_batches]
        out_batch.update({'clip_text': textbatch})

    if 'clip_path' in notnone_batches[0]:
        textbatch = [b['clip_path'] for b in notnone_batches]
        out_batch.update({'clip_path': textbatch})

    if 'all_categories' in notnone_batches[0]:
        textbatch = [b['all_categories'] for b in notnone_batches]
        out_batch.update({'all_categories': textbatch})

    return out_batch
