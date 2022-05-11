import os
import torch

from torch.utils.data import DataLoader
from src.utils.trainer import train
from src.utils.tensors import collate
import src.utils.fixseed  # noqa

from src.utils.get_model_and_data import get_model_and_data

from src.parser.checkpoint import parser


def add_epochs(model, datasets, parameters, optimizer, origepoch):
    dataset = datasets["train"]
    train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=8, collate_fn=collate)

    for epoch in range(1, parameters["num_epochs"]+1):
        dict_loss = train(model, optimizer, train_iterator, model.device)

        for key in dict_loss.keys():
            dict_loss[key] /= len(train_iterator)

        print(f"Epoch {epoch}, train losses: {dict_loss}")

        if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
            checkpoint_path = os.path.join(parameters["folder"],
                                           'retraincheckpoint_orig_{:04d}_added_{:04d}.pth.tar'.format(origepoch, epoch))
            print('Saving checkpoint {}'.format(checkpoint_path))
            torch.save(model.state_dict(), checkpoint_path)


def main():                
    # parse options
    parameters, folder, checkpointname, epoch = parser()    
    device = parameters["device"]
    
    model, datasets = get_model_and_data(parameters)
    datasets.pop("test")

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")
    add_epochs(model, datasets, parameters, optimizer, epoch)

    
if __name__ == '__main__':
    main()
