import torch

from torch import nn, Tensor
from Encoder import Encoder
from Decoder import Decoder
from LinearProjector import LinearProjector
from FramesDownSampler import FramesDownSampler


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

    def train(self, mode: bool = True):
        super().train(mode)

        # Only train decoder
        self.encoder.eval()

        # Only train Linear layers
        for name, param in self.decoder.named_parameters():
            if not any([linear_indicator in name for linear_indicator in ('fc', 'dense', 'linear')]):
                param.requires_grad = False

        return self

    def forward(
        self,
        input_values: Tensor,
        labels: Tensor = None,
        attention_mask: Tensor = None,
        input_length: Tensor = None,
    ) -> Tensor:

        speech_embeddings = self.encoder(input_values)

        down_sampled_speech_embeddings = self.down_sampler(speech_embeddings)

        projected_speech_embeddings = self.linear_projector(down_sampled_speech_embeddings)

        return self.decoder(speech_embeddings=projected_speech_embeddings, labels=labels)

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

if __name__ == "__main__":
    import torch

    model = SLAM()
    dummy_input = torch.randn((8, 258560))

    output = model(dummy_input)

    print(output['logits'].shape)
