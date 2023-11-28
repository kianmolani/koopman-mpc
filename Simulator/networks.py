""" 
Implementation of neural network modules required to learn, from data, our 
lifting dictionary and Koopman operator


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
        width (int)            :: width of hidden layers in encoder (as currently implemented, all hidden layers have the same width)
        output_dim (int)       :: width of output layer of encoder

        """

        super().__init__()

        self.input_dim = input_dim
        self.no_hidden_layers = no_hidden_layers
        self.width = width
        self.output_dim = output_dim

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
        all encoder layers are in an nn.Sequential container, the forward pass is
        computed automatically by calling self.encoder and passing in an input tensor

        x (torch.Tensor) :: input to encoder network

        return (torch.Tensor) :: output of encoder network

        """

        assert (x.shape[1]) == self.input_dim
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, no_hidden_layers, width, output_dim):
        """
        Initialization of decoder class

        input_dim (int)        :: width of input layer to decoder
        no_hidden_layers (int) :: number of hidden layers in decoder network
        width (int)            :: width of hidden layers in decoder (as currently implemented, all hidden layers have the same width)
        output_dim (int)       :: width of output layer of decoder

        """

        super().__init__()

        self.input_dim = input_dim
        self.no_hidden_layers = no_hidden_layers
        self.width = width
        self.output_dim = output_dim

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
        Returns output of forward pass through decoder network. Note that because
        all decoder layers are in an nn.Sequential container, the forward pass is
        computed automatically by calling self.decoder and passing in an input tensor

        x (torch.Tensor) :: input to decoder network

        return (torch.Tensor) :: output of decoder network

        """

        assert (x.shape[1]) == self.input_dim
        return self.decoder(x)


class Operator(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialization of operator class. The operator classes parameterizes the
        Koopman operator K, and represents action by the Koopman operator as a
        forward pass through a single linear-ReLu layer

        input_dim (int)  :: input dimension of operator network
        output_dim (int) :: output dimension of operator network

        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.ReLU())

        self.operator = nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns output of forward pass through operator network

        x (torch.Tensor) :: input to operator network

        return (torch.Tensor) :: output of operator network

        """

        assert (x.shape[1]) == self.input_dim
        return self.operator(x)


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, operator, linear_input_dim, linear_output_dim):
        """
        Initialization of predictor class. The predictor class puts together encoder,
        decoder, and operator networks in such a way so as to perform state predictions
        by action of the Koopman operator. After action by encoder, decoder, and
        operator networks, we pass the data through one final linear layer, as suggested
        by "Deep Learning for Universal Linear Embeddings of Nonlinear Dynamics" by
        Lusch et. al. For more information on implementation details, see
        https://www.nature.com/articles/s41467-018-07210-0 (architecture 'b')

        linear_input_dim (int)  :: input dimension of final linear layer
        linear_output_dim (int) :: output dimension of final output player

        """

        super().__init__()

        self.encoder = encoder
        self.operator = operator
        self.decoder = decoder
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)
        self.linear_input_dim = linear_input_dim
        self.linear_output_dim = linear_output_dim

    def forward(self, x):
        """
        Returns output of forward pass through predictor network

        x (torch.Tensor) :: input to predictor network

        return (torch.Tensor) :: output of predictor network

        """

        assert (x.shape[1]) == self.linear_input_dim
        return self.linear(self.decoder(self.operator(self.encoder(x))))
