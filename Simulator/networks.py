""" 
Implementation of neural network modules required to learn, from data, our 
lifting dictionary and Koopman operator. Complete architectures are built from
these modules but are constructed elsewhere (e.g. `track_circular_koopman.ipynb`)


Created by: Kian Molani
Last updated: Nov. 1, 2023

"""

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, no_hidden_layers, width, output_dim):
        """
        Initialization of encoder class

        input_dim (int)        :: width of input layer to encoder
        no_hidden_layers (int) :: number of hidden layers in encoder network
        width (int)            :: width of hidden layers in encoder. As currently implemented, all hidden layers have the same width
        output_dim (int)       :: number of output ...

        """
        super().__init__()

        layers = []

        # append first layer

        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())

        # append subsequent hidden layers

        for i in range(no_hidden_layers - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # append final layer

        layers.append(nn.Linear(width, output_dim))

        # create nn.Sequential object

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns output of forward pass through encoder network. Note that because
        all encoder layers are wrapped in a nn.Sequential container, the forward
        pass is computed automatically by just calling self.encoder

        x (torch.Tensor) :: arbitrary size torch.Tensor that represents inputs to the neural network

        return (list) :: list enumerating all 13 quadrotor states

        """

        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, no_hidden_layers, width, output_dim):
        """
        Initialization of decoder class

        input_dim (int)        :: width of input layer to decoder
        no_hidden_layers (int) :: number of hidden layers in decoder network
        width (int)            :: width of hidden layers in decoder. As currently implemented, all hidden layers have the same width

        """
        super().__init__()

        layers = []

        # append first layer

        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())

        # append subsequent hidden layers

        for i in range(no_hidden_layers - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # append final layer

        layers.append(nn.Linear(width, output_dim))

        # create nn.Sequential object

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns output of forward pass through encoder network. Note that because
        all encoder layers are wrapped in a nn.Sequential container, the forward
        pass is computed automatically by just calling self.encoder

        x (t)

        return (list) :: list enumerating all 13 quadrotor states

        """

        return self.decoder(x)


class Operator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.operator = nn.Linear(input_dim, output_dim)

    def forward(self, x, times):
        # for recursive calls
        return x if times <= 0 else self.forward(self.operator(x), times - 1)


class Predictor(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        operator,
        final_linear_input_dim,
        final_linear_output_dim,
    ):
        super().__init__()
        self.encoder = encoder
        self.operator = operator
        self.decoder = decoder
        self.linear = nn.Linear(final_linear_input_dim, final_linear_output_dim)

    def forward(self, x):
        return self.linear(self.decoder(self.operator(self.encoder(x))))
