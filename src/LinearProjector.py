from torch import nn, Tensor


class LinearProjector(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.linear_projector = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.linear_projector(t)
