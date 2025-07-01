"""A wrapper class for optimizer"""
import numpy as np


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, d_model, n_warmup_steps):
        """
        Initialize the ScheduledOptim class.

        Parameters:
        optimizer (torch.optim.Optimizer): The optimizer to wrap.
        d_model (int): The model dimension, used for scaling the learning rate.
        n_warmup_steps (int): The number of warmup steps for the learning rate schedule.
        """
        self.optimizer = optimizer  # Inner optimizer
        self.d_model = d_model  # Model dimension
        self.n_warmup_steps = n_warmup_steps  # Number of warmup steps
        self.n_current_steps = 0  # Initialize the current step count

    @property
    def param_groups(self):
        """
        Expose the param_groups of the wrapped optimizer.
        This is required for compatibility with external tools like GradScaler.
        """
        return self.optimizer.param_groups

    def step(self):
        """Step by the inner optimizer."""
        self.optimizer.step()  # Call the step method of the wrapped optimizer

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer."""
        self.optimizer.zero_grad()  # Call the zero_grad method of the wrapped optimizer

    def update_learning_rate(self):
        """Update the learning rate based on the current step."""
        self.n_current_steps += 1  # Increment the current step count

        # Calculate the new learning rate
        new_lr = (np.power(self.d_model, -0.5) *
                  np.min([
                      np.power(self.n_current_steps, -0.5),
                      np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
                  ]))

        # Update the learning rate for all parameter groups in the optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr  # Set the new learning rate for this parameter group
