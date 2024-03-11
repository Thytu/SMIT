import torch

from torch import nn, Tensor
from Encoder import Encoder
from Decoder import Decoder
from transformers import Wav2Vec2Processor
from LinearProjector import LinearProjector
from FramesDownSampler import FramesDownSampler
from transformers.modeling_outputs import CausalLMOutputWithPast


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
        self.processor = None

    def _init_processor(self):
        if self.processor is None:
            self.processor = Wav2Vec2Processor(
                feature_extractor=self.encoder.feature_extractor,
                tokenizer=self.decoder.tokenizer,
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

    def eval(self):
        return super().eval()

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
        self._init_processor()

        input_values = raw_speech
        # input_values = self.processor(
        #     raw_speech,
        #     sampling_rate=self.encoder.sampling_rate,
        # ).input_values[0]

        # input_features = [{"input_values": feature["input_values"]} for feature in features]

        # input_values = self.processor.pad(
        #     input_features,
        #     padding=True,
        #     # padding=self.padding_inputs,
        #     # max_length=self.max_length_inputs,
        #     return_tensors="pt",
        # )["input_values"]

        # print(input_values.shape)

        # TODO: new func on decoder calling self.decoder.greedy_search
        # https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/generation/utils.py#L1217

        outputs: CausalLMOutputWithPast = self.forward(input_values)

        next_token_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        new_tokens = self.decoder.tokenizer.convert_ids_to_tokens(next_token_ids.tolist())

        print(new_tokens)

        # TODO: continue autoregressively

        return new_tokens[0], next_token_ids[0]

# TODO LIST:
# SLAM generate_transcript method must be autoregressive to fully transcribe the input audio
# Write README


if __name__ == "__main__":
    import torch

    model = SLAM()
    dummy_input_values = torch.randn((8, 258560))

    output = model(dummy_input_values)

    print("Output:", output['logits'].shape)

    output = model.generate_transcript(dummy_input_values)

    print(output['logits'].shape)
