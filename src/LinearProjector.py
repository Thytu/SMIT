from torch import nn, Tensor


class LinearProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._layers = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim),
        )

    def forward(self, input_features: Tensor) -> Tensor:
        return self._layers(input_features)
