import torch
import json
import argparse
import os

import wandb
from pathlib import Path


from trainers.trainer_getter import get_trainer
from models.model_getter import get_model
from confs.conf_getter import get_conf
from datasets import get_dataloaders


def train(args):
    cfg = get_conf(args.conf)
    print(f'ARGS: {args}')
    print(f'PARAMS: {cfg}')

    cfg.model.dir = Path(f"./checkpoints/{args.conf}")

    wandb.init(project="normalizing-flows", name=args.conf, resume="allow", notes=f"{cfg}")

    os.makedirs(cfg.model.dir, exist_ok=True)

    model = get_model(cfg)
    if torch.cuda.is_available():
        model = model.cuda()


    train_dataloader, test_dataloader = get_dataloaders(cfg)

    trainer = get_trainer(cfg)
    trainer(cfg=cfg, model=model, train_dataset=train_dataloader, test_dataset=test_dataloader).train()


def get_parser():
    parser = argparse.ArgumentParser(description='train an neural network')
    parser.add_argument('--conf', default="flow_matching_cond", type=str)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    train(args)
