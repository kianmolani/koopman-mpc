from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, output_dim),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)