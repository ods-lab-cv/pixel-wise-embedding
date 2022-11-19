import torch
from collections import defaultdict
from tqdm.auto import tqdm


def make_class_mapping(dataset):
    mapping = defaultdict(list)
    for i, (image, mask) in enumerate(tqdm(dataset)):
        uniques = torch.unique(mask)
        for u in uniques:
            mapping[u].append(i)
    return mapping
