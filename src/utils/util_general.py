import argparse
from pathlib import Path
import os
import torch
import numpy as np
import random


def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()

def seed_all(seed):
    if not seed:
        seed = 0
    print("Using Seed : ", seed)

    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Empty and create directory
def create_dir(dir):
    if not os.path.exists(dir):
        Path(dir).mkdir(parents=True, exist_ok=True)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None, map_location=None):
    """

    Usabile:
    - TRAIN: load_checkpoint(path, model, optimizer=opt, lr=1e-4, map_location=device)
    - TEST:  load_checkpoint(path, model, map_location=device)

    """
    print("=> Loading checkpoint:", checkpoint_file)

    if map_location is not None:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
    else:
        checkpoint = torch.load(checkpoint_file)


    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)


    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr



