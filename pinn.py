import torch


def _construct_FC_layers(hidden, input_dim, output_dim, activations):
    """Constructs the layer structure for a fully connected neural network.
    returns list of layers
    """
    layers = []
    layers.append(torch.nn.Linear(input_dim, hidden[0]))
    layers.append(activations)
    for i in range(len(hidden) - 1):
        layers.append(torch.nn.Linear(hidden[i], hidden[i + 1]))
        layers.append(activations)
    layers.append(torch.nn.Linear(hidden[-1], output_dim))
    return layers


def normalize(
    x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor
) -> torch.Tensor:
    """scales [x_min, x_max] to [-1, 1] range.
        Parameters
    ----------
    x : torch.Tensor
        Input tensor to be normalized.
    x_min : torch.Tensor
        Minimum value(s) for normalization.
    x_max : torch.Tensor
        Maximum value(s) for normalization.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """
    return 2.0 * (x - x_min) / (x_max - x_min) - 1


class FCN(torch.nn.Module):
    """A simple fully connected neural network.

    Parameters
    ----------
    input_dim : int
        The space of the points the can be put into this model.
    output_dim : int
        The space of the points returned by this model.
    hidden : list or tuple
        The number and size of the hidden layers of the neural network.
        The lenght of the list/tuple will be equal to the number
        of hidden layers, while the i-th entry will determine the number
        of neurons of each layer.
        E.g hidden = (10, 5) -> 2 layers, with 10 and 5 neurons.
    activations : torch.nn, optional
        The activation functions of this network. Deafult is nn.Tanh().
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden=(20, 20, 20),
        activations=torch.nn.Tanh(),
        normalize=False,
        x_max=None,
        x_min=None,
    ):
        super().__init__()

        self.normalize = normalize
        if self.normalize:
            if x_min is None or x_max is None:
                raise ValueError(
                    "x_min and x_max must be provided when normalize is True."
                )
            self.register_buffer("x_max", torch.tensor(x_max).float())
            self.register_buffer("x_min", torch.tensor(x_min).float())

        layers = _construct_FC_layers(
            hidden=hidden,
            input_dim=input_dim,
            output_dim=output_dim,
            activations=activations,
        )
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, *x):
        x = torch.cat(x, dim=-1)  # concatenate along last dimension
        if self.normalize:
            x = normalize(x, self.x_min, self.x_max)
        return self.sequential(x)


def compute_gradients(f, x: torch.Tensor, q: torch.Tensor) -> tuple:
    """Computes gradient of NN wrt two variables. Assumes NN output is scalar.

    Returns:
    tuple of torch.Tensor: Gradients with respect to x and q.
    """
    x.requires_grad_(True)
    q.requires_grad_(True)

    # Forward pass
    y = f(x, q)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=(x, q),
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )
    return gradients


def uniform_sampler(batch_size: int, x_min, x_max) -> torch.Tensor:
    """
    Generates uniformly distributed samples within given ranges for each dimension.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        x_min : list, tuple, or torch.Tensor
            Lower bounds for each dimension.
        x_max : list, tuple, or torch.Tensor
            Upper bounds for each dimension.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, num_dimensions)` containing the uniformly sampled data.
    """
    x_min = torch.tensor(x_min)
    x_max = torch.tensor(x_max)

    # Generate uniform samples in the range [0, 1]
    rand_samples = torch.rand(batch_size, len(x_min))

    # Scale and shift the samples to the desired range
    samples = x_min + (x_max - x_min) * rand_samples

    return samples


def MS_loss_function(residual):
    return torch.mean(residual**2)


def smooth_max(inputs, dim, alpha=1000):
    return (torch.logsumexp(alpha * inputs, dim=dim)) / alpha


def smooth_abs(x, epsilon=1e-20):
    return torch.sqrt(x**2 + epsilon)
