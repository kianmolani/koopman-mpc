import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, no_hidden_layers, width):
        super().__init__()

        layers = []
        for i in range(no_hidden_layers):
            if i==0:
                layers.append(nn.Linear(input_dim, width))
                layers.append(nn.ReLU())
            elif i==no_hidden_layers-1:
                layers.append(nn.Linear(width, width))
            else:
                layers.append(nn.Linear(width, width))
                layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
    

class Decoder(nn.Module):
    def __init__(self, input_dim, no_hidden_layers, width):
        super().__init__()
        
        layers = []
        for i in range(no_hidden_layers):
            if i==0:
                layers.append(nn.Linear(input_dim, width))
                layers.append(nn.ReLU())
            elif i==no_hidden_layers-1:
                layers.append(nn.Linear(width, width))
            else:
                layers.append(nn.Linear(width, width))
                layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
    

class Operator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.operator = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.operator(x)


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, operator, final_linear_input_dim, final_linear_output_dim):
        super().__init__()
        self.encoder = encoder
        self.operator = operator
        self.decoder = decoder
        self.final_linear = nn.Linear(final_linear_input_dim, final_linear_output_dim)

    def forward(self, x):
        encoded_x = self.encoder(x)
        operator_output = self.operator(encoded_x)
        decoded_x = self.decoder(operator_output)
        output = self.final_linear(decoded_x)
        return output