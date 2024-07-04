from trainers.trainer import Trainer
import torch
import wandb
import torch.optim as optim
import pandas as pd



class FlowMatchingTrainer(Trainer):

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
        t = torch.rand(features.shape[0], 1, dtype=torch.float64)
        epsilon = self.pz.sample((features.shape[0],))
        y = t * features + (1 - t) * epsilon
        v_hat = self.model(y, t)
        v = features - epsilon  # y_1 - y_0
        loss = self.criterion(v_hat, v)
        return loss, v, None

    def run_valid_loop(self):
        self.model.eval()
        for i in range(3):
            samples = self.pz.sample((1000,))
            outputs = self.model.integrate(samples)
            df_samples = pd.DataFrame(outputs.detach().numpy(), columns=['x', 'y'])

            # Log the scatter plot to wandb
            wandb.log({
                f"scatter_plot_{i}_step_{self.step}": wandb.plot.scatter(wandb.Table(dataframe=df_samples), "x", "y", title=f"Custom Y vs X Scatter Plot {i}-{self.step}")
            }, step=self.step)


        torch.save(self.model.state_dict(), self.cfg.model.dir / f"epoch_{self.epoch}.pt")
        self.model.train()


