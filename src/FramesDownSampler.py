import torch

from torch import nn, Tensor


class FramesDownSampler(nn.Module):
    def __init__(self, k: int = 5) -> None:
        super().__init__()

        self.k = k

    def forward(self, frames: Tensor) -> Tensor:

        frames = frames[::,:frames.size(1) - frames.size(1) % self.k]

        chunks = torch.chunk(frames, frames.size(1) // self.k, dim=1)

        concatenated_frames = torch.stack(chunks).permute(1, 0, 2, 3)

        downsampled_frames = torch.sum(concatenated_frames, dim=2)

        return downsampled_frames


if __name__ == "__main__":
    import torch

    frames = torch.randn(1, 175, 1024)

    print("Input:", frames.shape)
    print("Output:", FramesDownSampler(k=5)(frames).shape)
