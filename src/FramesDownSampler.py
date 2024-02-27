import torch

from torch import nn, Tensor


class FramesDownSampler(nn.Module):
    def __init__(self, k: int = 5) -> None:
        super().__init__()
        
        self.k = k
        
    def forward(self, frames: Tensor) -> Tensor:
        chunks = torch.chunk(frames, frames.size(1) // self.k, dim=1)

        concatenated_frames = torch.cat(chunks, dim=0)

        down_sampled_frames = torch.sum(concatenated_frames, dim=1)
        
        return down_sampled_frames


if __name__ == "__main__":
    import torch

    frames = torch.randn(8, 20)
    
    print("Input:", frames.shape)
    print("Output:", FramesDownSampler(k=5)(frames).shape)
