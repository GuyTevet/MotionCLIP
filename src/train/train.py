import os
import sys
sys.path.append('.')

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from src.train.trainer import train
from src.utils.tensors import collate
import src.utils.fixseed  # noqa

from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data


def do_epochs(model, datasets, parameters, optimizer, writer):
    dataset = datasets["train"]
    train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=8, collate_fn=collate)

    logpath = os.path.join(parameters["folder"], "training.log")
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["num_epochs"]+1):
            dict_loss = train(model, optimizer, train_iterator, model.device)

            for key in dict_loss.keys():
                dict_loss[key] /= len(train_iterator)
                writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)

            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))
                print('Saving checkpoint {}'.format(checkpoint_path))
                state_dict_wo_clip = {k: v for k,v in model.state_dict().items() if not k.startswith('clip_model.')}
                torch.save(state_dict_wo_clip, checkpoint_path)

            writer.flush()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    # parse options
    parameters = parser()
    # parameters["device"] = "cuda:"
    # logging tensorboard
    writer = SummaryWriter(log_dir=parameters["folder"])

    device = parameters["device"] #"cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, datasets = get_model_and_data(parameters)

    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")

    # if device == "cpu":
    #     clip_model.float()
    # else:
    #     clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

    do_epochs(model, datasets, parameters, optimizer, writer)

    writer.close()
