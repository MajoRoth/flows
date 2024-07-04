import argparse

import numpy as np
import pandas as pd
import torch
import json
import argparse
import os
import wandb
from pathlib import Path

from confs.conf_getter import get_conf
from models.model_getter import get_model
import matplotlib.pyplot as plt
from tqdm import tqdm


def bonus(model, cfg):
    ckpt_path = cfg.model.dir / f"epoch_19.pt"
    model.load_state_dict(torch.load(ckpt_path))
    mean = torch.tensor([4.0, 5.0])

    # Standard deviation (you can adjust this value if needed)
    std_dev = torch.tensor([0.1, 0.1])

    # Generate 1000 points from a normal distribution
    points = mean + std_dev * torch.randn(1000, 2)
    noise = model.inverse_integrate(torch.tensor(points, dtype=torch.float64))
    noise_samples = pd.DataFrame(noise.detach().numpy(), columns=['x', 'y'])
    plt.scatter(noise_samples['x'],noise_samples['y'], s=1, color='red')

    recon_point = model.integrate(noise)

    df_samples = pd.DataFrame(recon_point.detach().numpy(), columns=['x', 'y'])

    plt.scatter(df_samples['x'],df_samples['y'], s=1, color='blue')
    plt.show()




def get_parser():
    parser = argparse.ArgumentParser(description='train an neural network')
    parser.add_argument('--conf', default="flow_matching_uncond", type=str)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    cfg = get_conf(args.conf)

    cfg.model.dir = Path(f"./../checkpoints/{args.conf}")

    # wandb.init(project="normalizing-flows-questions", name=args.conf, resume="allow", notes=f"{cfg}")

    os.makedirs(cfg.model.dir, exist_ok=True)

    model = get_model(cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    bonus(model, cfg)
