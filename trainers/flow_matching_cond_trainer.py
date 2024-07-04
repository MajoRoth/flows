from trainers.trainer import Trainer
import torch
import wandb
import torch.optim as optim
import pandas as pd
from tqdm import tqdm



class FlowMatchingCondTrainer(Trainer):

    def __init__(self, cfg, model, train_dataset, test_dataset):
        super().__init__(cfg, model, train_dataset, test_dataset)
        self.pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
        self.criterion = torch.nn.MSELoss()

    def train_step(self, features):
        return self.loss(features)

    def forward_and_loss(self, features):
        pass

    def loss(self, features):
        # sample t and epsilon
        points, labels = features
        t = torch.rand(points.shape[0], 1, dtype=torch.float64)
        epsilon = self.pz.sample((points.shape[0],))
        y = t * points + (1 - t) * epsilon
        v_hat = self.model(y, t, labels)
        v = points - epsilon  # y_1 - y_0
        loss = self.criterion(v_hat, v)
        return loss, v, None

    def run_valid_loop(self):
        self.model.eval()
        for cond in tqdm(range(5), desc=f'sampling'):
            samples = self.pz.sample((500,))
            outputs = self.model.integrate(samples, cond=torch.full((500,), cond))
            df_samples = pd.DataFrame(outputs.detach().numpy(), columns=['x', 'y'])

            # Log the scatter plot to wandb
            wandb.log({
                f"scatter_cond_{cond}_step_{self.step}": wandb.plot.scatter(wandb.Table(dataframe=df_samples), "x", "y", title=f"Custom Y vs X Scatter Plot cond_{cond}-{self.step}")
            }, step=self.step)


        torch.save(self.model.state_dict(), self.cfg.model.dir / f"epoch_{self.epoch}.pt")
        self.model.train()


