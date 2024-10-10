import torch
from torch import nn
from torch.distributions import Normal


class VIDREncoder(nn.Module):
    '''  Variational Encoder for scVIDR (Single-Cell Variational Inference for Dose Response).

    This class implements the encoder portion of a variational autoencoder (VAE),
    which encodes input data into a latent representation by learning mean and
    variance for reparameterization.

    Attributes:
        eps (float): Small constant added to variance for numerical stability.
        fclayers (nn.Sequential): Fully connected layers with batch normalization,
            dropout, and activation.
        encoder (nn.Sequential): The full encoder model consisting of input and hidden layers.
        mean (nn.Linear): Linear layer to compute the mean of the latent distribution.
        log_var (nn.Linear): Linear layer to compute the log-variance of the latent distribution.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent representation.
        hidden_dim (int): Number of hidden units in the fully connected layers.
        n_hidden_layers (int): Number of hidden layers.
        momentum (float, optional): Momentum parameter for batch normalization. Defaults to 0.01.
        eps (float, optional): Epsilon for numerical stability in batch normalization. Defaults to 0.001.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        reparam_eps (float, optional): Epsilon added to variance during reparameterization. Defaults to 1e-4.
        nonlin (callable, optional): Non-linear activation function. Defaults to nn.LeakyReLU.

    Methods:
        forward(inputs):
            Encodes the input data, computes mean and variance, and performs reparameterization.
    '''
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        momentum: float = 0.01,
        eps: float = 0.001,
        dropout_rate: float = 0.2,
        reparam_eps: float = 1e-4,
        nonlin=nn.LeakyReLU,
    ):
        super(VIDREncoder, self).__init__()
        self.eps = reparam_eps
        self.fclayers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
            nonlin(),
            nn.Dropout(p=dropout_rate),
        )

        # Encoder
        # Include Input Module
        modules = [
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
                nonlin(),
                nn.Dropout(p=dropout_rate),
            )
        ]

        # Add hidden fully connected layers
        for i in range(n_hidden_layers - 1):
            modules.append(self.fclayers)

        self.encoder = nn.Sequential(*modules)

        self.mean = nn.Linear(hidden_dim, latent_dim)

        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs):
        ''' Forward pass of the encoder.
        
        Encodes the input data into a latent representation by
        computing the mean and variance, then sampling using 
        the reparameterization trick.

        Args:
            inputs (torch.Tensor): The input data.

        Returns:
            tuple: A tuple containing:
                - mean (torch.Tensor): The mean of the latent distribution.
                - var (torch.Tensor): The variance of the latent distribution.
                - latent_rep (torch.Tensor): The reparameterized latent representation.

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        '''
        # encode
        results = self.encoder(inputs)
        mean = self.mean(results)
        log_var = self.log_var(results)

        var = torch.exp(log_var) + self.eps

        # reparameterize
        latent_rep = Normal(mean, var.sqrt()).rsample()

        return mean, var, latent_rep


class VIDRDecoder(nn.Module):
    ''' Variational Decoder for scVIDR (Single-Cell Variational Inference for Dose Response).
    
    This class implements the decoder portion of a variational autoencoder (VAE),
    which decodes latent representations back into input space.

    Attributes:
        fclayers (nn.Sequential): Fully connected layers with batch normalization,
            dropout, and activation.
        decoder (nn.Sequential): The full decoder model consisting of latent input, hidden layers,
            and output layer.

    Args:
        input_dim (int): Dimensionality of the output data.
        latent_dim (int): Dimensionality of the latent representation.
        hidden_dim (int): Number of hidden units in the fully connected layers.
        n_hidden_layers (int): Number of hidden layers.
        momentum (float, optional): Momentum parameter for batch normalization. Defaults to 0.01.
        eps (float, optional): Epsilon for numerical stability in batch normalization. Defaults to 0.001.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        nonlin (callable, optional): Non-linear activation function. Defaults to nn.LeakyReLU.

    Methods:
        forward(latent_rep):
            Decodes the latent representation back into the input space.
    '''
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        momentum: float = 0.01,
        eps: float = 0.001,
        dropout_rate: float = 0.2,
        nonlin=nn.LeakyReLU,
    ):
        super(VIDRDecoder, self).__init__()
        self.fclayers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
            nonlin(),
            nn.Dropout(p=dropout_rate),
        )

        # Encoder
        # Include Input Module
        modules = [
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
                nonlin(),
                nn.Dropout(p=dropout_rate),
            )
        ]

        # Add hidden fully connected layers
        for i in range(n_hidden_layers - 1):
            modules.append(self.fclayers)

        modules.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*modules)

    def forward(self, latent_rep):
        ''' Forward pass of the decoder.
        
        Decodes the latent representation back into
        the original input space.

        Args:
            latent_rep (torch.Tensor): The latent representation to decode.

        Returns:
            torch.Tensor: The reconstructed data.
        '''
        # decode
        x_hat = self.decoder(latent_rep)
        return x_hat
