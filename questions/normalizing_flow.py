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


def sample(model, cfg):
    ckpt_path = cfg.model.dir / f"epoch_19.pt"
    model.load_state_dict(torch.load(ckpt_path))

    pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    samples = pz.sample((1000,))
    outputs = model(samples)

    df_samples = pd.DataFrame(outputs.detach().numpy(), columns=['x', 'y'])

    # Log the scatter plot to wandb
    wandb.log({
        f"scatter_plot": wandb.plot.scatter(wandb.Table(dataframe=df_samples), "x", "y",
                                            title=f"Custom Y vs X Scatter Plot")
    })


def q3(model, cfg):
    ckpt_path = cfg.model.dir / f"epoch_19.pt"
    model.load_state_dict(torch.load(ckpt_path))

    pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    samples = pz.sample((1000,))

    x = samples
    for i, layer in enumerate(model.layers):
        x = layer(x)

        if i % 6 == 0:
            df_samples = pd.DataFrame(x.detach().numpy(), columns=['x', 'y'])

            # Log the scatter plot to wandb
            wandb.log({
                f"q3 layer {i}": wandb.plot.scatter(wandb.Table(dataframe=df_samples), "x", "y",
                                                    title=f"Custom Y vs X Scatter Plot q3 layer {i}")
            })

    df_samples = pd.DataFrame(x.detach().numpy(), columns=['x', 'y'])

    # Log the scatter plot to wandb
    wandb.log({
        f"q3 layer {i}": wandb.plot.scatter(wandb.Table(dataframe=df_samples), "x", "y",
                                            title=f"Custom Y vs X Scatter Plot q3 layer {i}")
    })


def q4(model, cfg):
    ckpt_path = cfg.model.dir / f"epoch_19.pt"
    model.load_state_dict(torch.load(ckpt_path))

    pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    samples = pz.sample((10,))

    x = samples
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0, 1, 16))  # Generate 10 colors from Viridis colormap

    df_samples = pd.DataFrame(x.detach().numpy(), columns=['x', 'y'])
    plt.scatter(df_samples['x'], df_samples['y'], color=colors[0], label=f'p(z)')

    for i, layer in enumerate(model.layers):
        x = layer(x)

        if i % 2 == 0:
            df_samples = pd.DataFrame(x.detach().numpy(), columns=['x', 'y'])

            # Create scatter plot
            plt.scatter(df_samples['x'], df_samples['y'], color=colors[i // 2 + 1], label=f'Layer {i//2}')


    df_samples = pd.DataFrame(x.detach().numpy(), columns=['x', 'y'])

    plt.scatter(df_samples['x'], df_samples['y'], color=colors[15], label=f'Final Layer')

    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(bbox_to_anchor=(1.25, 0.5), loc='center left')
    plt.tight_layout()

    plt.show()



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

    df_samples = pd.DataFrame(y.detach().numpy(), columns=['x', 'y'])
    for j in range(5):
        r, g, b = point_colors[j]  # Cycle through the colors for each point
        plt.scatter(df_samples['x'][j], df_samples['y'][j], color=(r, g, b, 1))

    for i, layer in enumerate(reversed(model.layers)):
        y = layer.inverse(y)
        df_samples = pd.DataFrame(y.detach().numpy(), columns=['x', 'y'])

        if i%4 == 0:
            for j in range(5):
                r, g, b = point_colors[j]  # Cycle through the colors for each point
                plt.scatter(df_samples['x'][j], df_samples['y'][j], color=[(r, g, b, 0.7 - 0.02*i)])

    df_samples = pd.DataFrame(y.detach().numpy(), columns=['x', 'y'])
    for j in range(5):
        r, g, b = point_colors[j]  # Cycle through the colors for each point
        plt.scatter(df_samples['x'][j], df_samples['y'][j], color=[(r, g, b, 0.8 - 0.02 * i)])

    p = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    p.log_prob(x)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()


def get_parser():
    parser = argparse.ArgumentParser(description='train an neural network')
    parser.add_argument('--conf', default="normalizing_flow", type=str)
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
