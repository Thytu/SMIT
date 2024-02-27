import torch

from torch import nn, Tensor
from .Encoder import Encoder
from .Decoder import Decoder
from .LinearProjector import LinearProjector
from .FramesDownSampler import FramesDownSampler


class SLAM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = Encoder(sampling_rate=16_000)
        self.down_sampler = FramesDownSampler(k=5)
        self.decoder = Decoder(model_name="microsoft/phi-2")

        self.linear_projector = LinearProjector(
            input_dim=self.encoder.output_dim,
            output_dim=self.decoder.model.config.hidden_size,
        )

    # TODO: it must accept batches!
    # currently it takes as an input a single array representing a single audio sample
    def forward(self, raw_speech: Tensor) -> Tensor:

        speech_embeddings = self.encoder(raw_speech)

        down_sampled_speech_embeddings = self.down_sampler(speech_embeddings)

        projected_speech_embeddings = self.linear_projector(down_sampled_speech_embeddings)

        logits = self.decoder(speech_embeddings=projected_speech_embeddings)

        return logits

    def generate_transcript(self, raw_speech):
        logits = self.forward(raw_speech)

        next_token_ids = torch.argmax(logits[:, -1, :], dim=-1)

        new_tokens = self.decoder.tokenizer.convert_ids_to_tokens(next_token_ids.tolist())

        # TODO: continue autoregressively

        return new_tokens[0], next_token_ids[0]

# TODO LIST:
# SLAM must takes a batched tensor as input and not a list representing a single audio sample
# write data preprocessing
# train the model using Trainer + link it to WnB
# SLAM generate_transcript method must be autoregressive to fully transcribe the input audio
# Clean this file
# Write README
