from .amass import AMASS

def get_dataset(name="amass"):
    return AMASS


def get_datasets(parameters, clip_preprocess, split="train"):
    DATA = AMASS

    if split == 'all':
        train = DATA(split='train', clip_preprocess=clip_preprocess, **parameters)
        test = DATA(split='vald', clip_preprocess=clip_preprocess, **parameters)

        # add specific parameters from the dataset loading
        train.update_parameters(parameters)
        test.update_parameters(parameters)
    else:
        dataset = DATA(split=split, clip_preprocess=clip_preprocess, **parameters)
        train = dataset

        # test: shallow copy (share the memory) but set the other indices
        from copy import copy
        test = copy(train)
        test.split = test

        # add specific parameters from the dataset loading
        dataset.update_parameters(parameters)

    datasets = {"train": train,
                "test": test}

    return datasets
