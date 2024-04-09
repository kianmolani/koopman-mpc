""" 
Implementation of neural network modules required to learn, from data, our 
lifting dictionary and linear ("Koopman") operators


Created by: Kian Molani
Last updated: Feb. 22, 2024

"""

import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, no_hidden_layers, width, output_dim):
        """
        Initialization of encoder class. The task of the encoder network is
        to lift our state-space vector to a higher dimensional meta-space of
        functions of our state

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
        Returns output of forward pass through encoder network

        x (torch.Tensor) :: input to encoder network

        return (torch.Tensor) :: output of encoder network

        """

        assert (x.shape[1]) == self.input_dim
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, no_hidden_layers, width, output_dim):
        """
        Initialization of decoder class. The task of the decoder network is
        to bring our meta-space vector back down to state-space

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
        Returns output of forward pass through decoder network

        x (torch.Tensor) :: input to decoder network

        return (torch.Tensor) :: output of decoder network

        """

        assert (x.shape[1]) == self.input_dim
        return self.decoder(x)
    

class Operator(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialization of operator class. The operator class can parameterize
        any linear operator, such as the 'A' operator which is associated to 
        action on the lifted state. Linear operators are parameterized here by 
        a single linear layer of neurons. Note that this linear layer is not 
        appended by an activation layer because the action of a linear layer 
        is equivalent (in a mathematical) to the action of a linear operator.
        For the same reason, we do not have any node biases

        input_dim (int)  :: input dimension of operator network
        output_dim (int) :: output dimension of operator network

        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        layers.append(nn.Linear(input_dim, output_dim, bias=False))

        self.operator = nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns output of forward pass through operator network

        x (torch.Tensor) :: input to operator network

        return (torch.Tensor) :: output of operator network

        """

        assert (x.shape[1]) == self.input_dim
        return self.operator(x)


class Adder(nn.Module):
    def __init__(self, *operators):
        """
        Initialization of adder class. Given an arbitrary number of linear 
        operators, the adder class captures the following computation: e.g.
        A * z + B * x + C * u. Summation in neural networks can be achieved
        by passing e.g. the outputs A * z and B * x (both computed in parallel,
        not sequentially) to a final linear layer with no biases and fixed
        weights set as the concatenation of identity matrices. We assume here 
        that the first operator in the list of operators acts on our lifted 
        state vector. We make no assumptions on the rest of the operators

        operators (tuple) :: tuple of Operator class objects, where each operator is a neural network parameterizing a linear operator acting on some input

        """
        
        super().__init__()

        no_operators = len(operators)
        self.operators = operators
        self.meta_space_dim = self.operators[0].operator[0].out_features
        self.final_linear_layer = nn.Linear(self.meta_space_dim*no_operators, self.meta_space_dim, bias=False)
        self.weight_tensor = torch.tensor(np.concatenate([np.eye(self.meta_space_dim)] * no_operators, axis=1), dtype=torch.float32)

        # set and fix weights of network's final linear layer

        self.final_linear_layer.weight = nn.Parameter(self.weight_tensor, requires_grad=False)

    def forward(self, *x):
        """
        Returns output of forward pass through adder network

        x (torch.Tensor) :: tuple of torch input tensors (e.g. lifted state vector, state space vector, control input vector)

        return (torch.Tensor) :: output of adder network

        """

        # assert shape and comment

        assert (x[0].shape[1]) == self.meta_space_dim
        return self.final_linear_layer(torch.cat((self.operators[0](x[0]), self.operators[1](x[1]), self.operators[2](x[2])), dim=1))
    

class Predictor(nn.Module):
    def __init__(self, encoder, adder, decoder, linear_input_dim, linear_output_dim):
        """
        Initialization of predictor class. The predictor class puts together
        encoder, decoder, and adder networks in such a way so as to perform
        dynamics predictions of state-space vectors. After action by encoder, 
        decoder, and adder networks, we pass the data through one final linear 
        layer as suggested by https://www.nature.com/articles/s41467-018-07210-0. 
        For more information on implementation details, please refer to 
        architecture 'b' in this paper. Note, however, that this architecture 
        has been extended here to the controlled setting

        encoder (Encoder)       :: Encoer class object
        adder (Adder)           :: Adder class object
        decoder (Decoder)       :: Decoder class object
        linear_input_dim (int)  :: input dimension of final linear layer
        linear_output_dim (int) :: output dimension of final output player

        """

        super().__init__()

        self.encoder = encoder
        self.adder = adder
        self.decoder = decoder
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)
        self.linear_input_dim = linear_input_dim
        self.linear_output_dim = linear_output_dim

    def forward(self, *x):
        """
        Returns output of forward pass through predictor network. Assumes 
        first element of tuple lifted state vector

        x (torch.Tensor) :: input representing lifted state vector
        u (torch.Tensor) :: input representing control input vector

        return (torch.Tensor) :: output of predictor network

        """

        assert (x[0].shape[1]) == self.linear_input_dim
        return self.linear(self.decoder(self.adder(self.encoder(x[0]), x[1], x[2])))