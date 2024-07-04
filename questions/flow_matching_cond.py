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

colors = ['black', 'blue', 'green', 'red', 'yellow']


def sample(model, cfg):
    ckpt_path = cfg.model.dir / f"epoch_19.pt"
    model.load_state_dict(torch.load(ckpt_path))

    pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    for i in tqdm(range(5)):
        samples = pz.sample((1000,))
        conditions = torch.randint(0, 5, (1000,))
        outputs = model.integrate(samples, conditions)

        df_samples = pd.DataFrame(outputs.detach().numpy(), columns=['x', 'y'])

        condition_colors = [colors[cond] for cond in conditions.numpy()]
        plt.scatter(df_samples['x'],df_samples['y'], s=1, c=np.array(condition_colors))

    plt.show()


def q2(model, cfg):
    ckpt_path = cfg.model.dir / f"epoch_19.pt"
    model.load_state_dict(torch.load(ckpt_path))
    colors = ['black', 'blue', 'green', 'red', 'yellow']

    pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    samples = pz.sample((5,))
    conditions = torch.tensor([0, 1, 2, 3, 4])

    for t in tqdm(np.linspace(0, 1, 70)):
        outputs = model.integrate(samples.clone(), cond=conditions.clone(), t_max=t)
        df_samples = pd.DataFrame(outputs.detach().numpy(), columns=['x', 'y'])

        plt.scatter(df_samples['x'], df_samples['y'], s=10, c=colors, alpha=t)

    plt.show()


def get_parser():
    parser = argparse.ArgumentParser(description='train an neural network')
    parser.add_argument('--conf', default="flow_matching_cond", type=str)
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

    sample(model, cfg)
    # q2(model, cfg)
