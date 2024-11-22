from typing import List, Tuple, Optional, Dict, Any
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import math
import pandas as pd


from pinn import (
    uniform_sampler,
    smooth_abs,
    MS_loss_function,
    FCN,
)
from physics import feq, BE_residue


class BoltzmaNN:
    def __init__(
        self,
        x_range: Tuple[float, float],
        q_range: Tuple[float, float],
        hidden_width: int,
        batch_size: int,
        NumericalSolution: list,
        x_lin_num: np.ndarray,
        q_lin_num: np.ndarray,
        collocation_number_plot=200,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x0, self.xf = x_range
        self.q0, self.qf = q_range
        self.NumericalSolution = NumericalSolution
        self.NumericalFinalDistribution = NumericalSolution[-1, :]
        self.NumericalInitialDistribution = NumericalSolution[0, :]
        self.q_lin_num = q_lin_num
        self.x_lin_num = x_lin_num

        # Initialize model, optimizer, scheduler, and other settings
        self.model = FCN(
            input_dim=2,
            output_dim=1,
            hidden=(hidden_width, hidden_width),
            normalize=True,
            x_min=[self.x0, self.q0],
            x_max=[self.xf, self.qf],
            activations=torch.nn.Tanh(),
        )

        # Training parameters
        self.batch_size = batch_size
        self.collocation_number_plot = collocation_number_plot
        self.plotting_step = 1000
        # self.resampling_interval = 100

        # History trackers
        self.loss_history = []
        self.MSE_numerical = np.array([])
        self.history_distributions_final = []
        self.history_distributions_initial = []

        # Plotting variables
        self.q_values = (
            torch.linspace(self.q0, self.qf, steps=collocation_number_plot)
            .view(collocation_number_plot, 1)
            .to(self.device)
        )
        self.x_values = (
            torch.linspace(self.x0, self.xf, steps=collocation_number_plot)
            .view(collocation_number_plot, 1)
            .to(self.device)
        )
        self.x_values_fin = torch.ones_like(self.q_values) * self.xf
        self.x_values_ini = torch.ones_like(self.q_values) * self.x0
        self.q_value_max = torch.ones_like(self.q_values) * 2
        self.q_values.requires_grad = True
        self.x_values_fin.requires_grad = True
        self.x_values_ini.requires_grad = True
        self.x_values.requires_grad = True
        self.q_value_max.requires_grad = True
        self.x_plot_values = self.x_values.cpu().detach().numpy()
        self.q_plot_values = self.q_values.cpu().detach().numpy()

    def sample_points(self, q_min, q_max, x_min, x_max):
        x_samples = uniform_sampler(self.batch_size, [x_min], [x_max])
        q_samples = uniform_sampler(self.batch_size, [q_min], [q_max])
        x_grid, q_grid = torch.meshgrid(
            x_samples.squeeze(), q_samples.squeeze(), indexing="ij"
        )
        xq_meshgrid = torch.stack((x_grid, q_grid), dim=-1).reshape(-1, 2)
        xq_meshgrid.requires_grad = True
        return xq_meshgrid[:, 0].view(-1, 1).to(self.device), xq_meshgrid[:, 1].view(
            -1, 1
        ).to(self.device)

    def train(
        self,
        num_epochs,
        physics_weight,
        positivity_weight,
        bc_weight,
        initialC_weight,
        learning_rate=1e-2,
        patience=800,
        optimizer=torch.optim.Adam,
        scheduler=ReduceLROnPlateau,
        **kwargs,
    ):
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, **kwargs)

        self.scheduler = scheduler(
            self.optimizer,
            factor=0.5,
            patience=patience,
            min_lr=1e-8,
            threshold=1e-5,
            eps=0,
        )

        self.model.train().to(self.device)
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            x_grid, q_grid = self.sample_points(self.q0, self.qf, self.x0, self.xf)

            # Compute boundary condition losses
            boundary_loss_start = MS_loss_function(
                self.model(x_grid, torch.zeros_like(q_grid))
            )
            boundary_loss_end = MS_loss_function(
                self.model(
                    x_grid, self.qf * torch.ones_like(q_grid)
                )  # - qf**2 / (np.exp(qf) - 1)
            )
            bc_loss = bc_weight * (boundary_loss_start + boundary_loss_end)

            # Compute initial condition loss
            initial_condition_loss = initialC_weight * MS_loss_function(
                self.model(self.x0 * torch.ones_like(x_grid), q_grid) - feq(q_grid)
            )

            # Adaptive physics weight for learning IC first
            current_physics_weight = physics_weight  # * (epoch / num_epochs)
            # Compute physics loss
            equation_residual_loss = current_physics_weight * MS_loss_function(
                BE_residue(self.model, x_grid, q_grid)
            )

            # Compute positivity constraint
            positivity_loss = positivity_weight * MS_loss_function(
                smooth_abs(self.model(x_grid, q_grid)) - self.model(x_grid, q_grid)
            )

            # Total loss and backward pass
            total_loss = (
                equation_residual_loss
                + bc_loss
                + initial_condition_loss
                + positivity_loss
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scheduler.step(total_loss)
            self.optimizer.step()

            # Track losses
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.loss_history.append(
                {
                    "boundary": bc_loss.item(),
                    "initial": initial_condition_loss.item(),
                    "physics": equation_residual_loss.item(),
                    "positivity": positivity_loss.item(),
                    "weighted_sum": total_loss.item(),
                    "last LR": current_lr,
                }
            )

            if epoch % self.plotting_step == 0 and epoch != 0:
                print(self.loss_history[-1])
                self.update_mse_numerical(epoch)
                self.record_distribution_history()

        self.loss_history = pd.DataFrame(self.loss_history)
        return self.MSE_numerical[-1], total_loss

    def update_mse_numerical(self, epoch):
        x_lin_tensor = torch.from_numpy(self.x_lin_num).float()
        q_lin_tensor = torch.from_numpy(self.q_lin_num).float()

        # Corrected variable names in meshgrid function
        x_grid, q_grid = torch.meshgrid(x_lin_tensor, q_lin_tensor, indexing="ij")

        # Stack and reshape the meshgrid to create a list of (x, q) pairs
        xq_meshgrid = torch.stack((x_grid, q_grid), dim=-1).reshape(-1, 2)

        # Enable gradient computation
        xq_meshgrid.requires_grad_(True)

        # Split the meshgrid back into x and q tensors
        x_lin_grid = xq_meshgrid[:, 0].view(-1, 1).to(self.device)
        q_lin_grid = xq_meshgrid[:, 1].view(-1, 1).to(self.device)

        predicted_final_distribution = (
            self.model(x_lin_grid, q_lin_grid).cpu().detach().numpy().reshape(-1)
        )
        predicted_final_distribution = predicted_final_distribution.reshape(
            x_grid.shape
        )

        rmse = np.sqrt(
            np.mean((self.NumericalSolution - predicted_final_distribution) ** 2)
        )
        self.MSE_numerical = np.append(self.MSE_numerical, rmse)
        print(f"Epoch {epoch}: MSE wrt numerical = {rmse}")

    def record_distribution_history(self):
        predicted_final = self.model(self.x_values_fin, self.q_values).cpu().detach()
        predicted_initial = self.model(self.x_values_ini, self.q_values).cpu().detach()
        self.history_distributions_final.append(predicted_final)
        self.history_distributions_initial.append(predicted_initial)

    def plot_steps(self):
        num_plots = len(self.history_distributions_final)
        rows = math.ceil(num_plots / 2)  # number of rows needed
        fig, axes = plt.subplots(
            rows, 2, figsize=(10, 4 * rows)
        )  # Adjust the figure size for readability

        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

        for epoch_index, (predicted_final, predicted_initial) in enumerate(
            zip(self.history_distributions_final, self.history_distributions_initial)
        ):
            ax = axes[epoch_index]
            ax.plot(self.q_plot_values, predicted_final, label="PINN final", color="r")
            ax.plot(
                self.q_plot_values, predicted_initial, label="PINN initial", color="g"
            )
            ax.plot(
                self.q_lin_num,
                self.NumericalFinalDistribution,
                "r--",
                label="Numerical, final",
            )
            ax.plot(
                self.q_lin_num,
                self.NumericalInitialDistribution,
                "g--",
                label="Numerical, initial",
            )
            ax.set_xlabel("q")
            ax.set_ylabel("q^2 f(q)")
            ax.set_title(f"#epochs={(epoch_index+1) * self.plotting_step}")
            ax.legend()

        # Hide unused subplots if any
        for i in range(epoch_index + 1, len(axes)):
            axes[i].axis("off")

        plt.suptitle("PINN vs Numerical", y=1.01)
        plt.tight_layout()
        plt.show()

    def plot_last(self, n):
        # Ensure n is not larger than the available history
        n = min(n, len(self.history_distributions_final))

        for i in range(-n, 0):  # Loop over the last n distributions
            alpha_paling = 1 + (1 / n) * (1 + i)
            plt.plot(
                self.q_plot_values,
                self.history_distributions_final[i],
                label=f"PINN final (Epoch {(len(self.history_distributions_final)+1 + i)*self.plotting_step})",
                color="r",
                alpha=alpha_paling,
            )
            plt.plot(
                self.q_plot_values,
                self.history_distributions_initial[i],
                label=f"PINN initial (Epoch {(len(self.history_distributions_initial)+1 + i)*self.plotting_step})",
                color="g",
                alpha=alpha_paling,
            )

        plt.plot(
            self.q_lin_num,
            self.NumericalFinalDistribution,
            "r--",
            label="Numerical, final",
        )
        plt.plot(
            self.q_lin_num,
            self.NumericalInitialDistribution,
            "g--",
            label="Numerical, initial",
        )

        # Set labels and title
        plt.xlabel("q")
        plt.ylabel("q^2 f(q)")
        plt.title(
            f"Last {n} Epochs (#epochs={len(self.history_distributions_final) * self.plotting_step})"
        )

        # Show legend and plot
        plt.legend()
        plt.show()

    def plot_learning_data(self, num_epochs):
        Loss_of_q = (
            torch.mean(
                BE_residue(self.model, self.x_values_fin, self.q_values) ** 2, dim=1
            )
            .cpu()
            .detach()
            .numpy()
        )
        Loss_of_x = (
            torch.mean(
                BE_residue(self.model, self.x_values, self.q_value_max) ** 2, dim=1
            )
            .cpu()
            .detach()
            .numpy()
        )

        _, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(self.q_plot_values, Loss_of_q, "c", label="Physics loss at xf in q")
        ax[1].plot(self.x_plot_values, Loss_of_x, "m", label="Physics loss at q=2 in x")
        for a in ax:
            a.legend()
        plt.tight_layout()
        plt.show()

        self.plot_error(num_epochs)

    def plot_error(self, num_epochs):
        plt.plot(
            np.linspace(0, num_epochs, self.MSE_numerical.size),
            self.MSE_numerical,
            label="MSE wrt numerical",
        )
        plt.legend()
        plt.show()
