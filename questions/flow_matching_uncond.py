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


def q2(model, cfg):
    ckpt_path = cfg.model.dir / f"epoch_19.pt"
    model.load_state_dict(torch.load(ckpt_path))

    pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))

    for t in tqdm([0, 0.2, 0.4, 0.6, 0.8, 1]):
        samples = pz.sample((1000,))

        outputs = model.integrate(samples, t_max=t)

        df_samples = pd.DataFrame(outputs.detach().numpy(), columns=['x', 'y'])

        # Log the scatter plot to wandb
        wandb.log({
            f"scatter_plot_t_{t}": wandb.plot.scatter(wandb.Table(dataframe=df_samples), "x", "y",
                                                title=f"Custom Y vs X Scatter Plot t={t}")
        })


def q3(model, cfg):
    colors = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black']

    ckpt_path = cfg.model.dir / f"epoch_19.pt"
    model.load_state_dict(torch.load(ckpt_path))

    pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    samples = pz.sample((10,))

    for t in tqdm(np.linspace(0, 1, 70)):
        outputs = model.integrate(samples.clone(), t_max=t)
        df_samples = pd.DataFrame(outputs.detach().numpy(), columns=['x', 'y'])

        plt.scatter(df_samples['x'], df_samples['y'], s=10, c=[(0.2, 0.2, 0.2, 0.3 + 0.7*t) for p in samples])

    plt.show()


def q4(model, cfg):
    ckpt_path = cfg.model.dir / f"epoch_19.pt"
    model.load_state_dict(torch.load(ckpt_path))

    pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))

    for dt in tqdm([0.002, 0.02, 0.05, 0.1, 0.2]):
        samples = pz.sample((1000,))
        outputs = model.integrate(samples, dt=dt)

        df_samples = pd.DataFrame(outputs.detach().numpy(), columns=['x', 'y'])

        # Log the scatter plot to wandb
        wandb.log({
            f"scatter_plot_dt_{dt}": wandb.plot.scatter(wandb.Table(dataframe=df_samples), "x", "y",
                                                title=f"Custom Y vs X Scatter Plot dt={dt}")
        })

#
# def sample(model, cfg):
#     ckpt_path = cfg.model.dir / f"epoch_19.pt"
#     model.load_state_dict(torch.load(ckpt_path))
#
#     pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
#     for i in tqdm(range(5)):
#         samples = pz.sample((1000,))
#         conditions = torch.randint(0, 5, (1000,))
#         outputs = model.integrate(samples, conditions)
#
#         df_samples = pd.DataFrame(outputs.detach().numpy(), columns=['x', 'y'])
#
#         condition_colors = [colors[cond] for cond in conditions.numpy()]
#         plt.scatter(df_samples['x'],df_samples['y'], s=1, c=np.array(condition_colors))
#
#     plt.show()


def q5(model, cfg):
    points_in = [
        (0, -0.6),
        (0.6, 0.6),
        (-0.6, 0.6)
    ]

    points_out = [
        (-2, 2.4),
        (0.7, 2)
    ]

    ckpt_path = cfg.model.dir / f"epoch_19.pt"
    model.load_state_dict(torch.load(ckpt_path))

    y = torch.tensor(points_out + points_in, dtype=torch.float64)
    # sum_log_det = torch.zeros(y.shape[0])
    point_colors = [(0.0, 0.0, 1.0),  # blue
                    (0.0, 0.7, 1.0),  # blue
                    (1.0, 0.0, 0.0),  # red
                    (1.0, 0.0, 0.3),  # red
                    (1.0, 0.3, 0.0)]  # red


    with torch.no_grad():
        for time in np.arange(1, 0, -0.03):
            y -= model.forward(y, torch.full((y.shape[0], 1), time)) * 0.03

            df_samples = pd.DataFrame(y.detach().numpy(), columns=['x', 'y'])
            for j in range(5):
                r, g, b = point_colors[j]  # Cycle through the colors for each point
                if time == 1:
                    r, g, b = (0, 0, 0)
                    plt.scatter(df_samples['x'][j], df_samples['y'][j], color=[(r, g, b, 1)])
                else:
                    plt.scatter(df_samples['x'][j], df_samples['y'][j], color=[(r, g, b, 0.5* time)])


    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
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

    # sample(model, cfg)
    q5(model, cfg)
