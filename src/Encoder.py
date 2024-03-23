from torch import nn, Tensor
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class Encoder(nn.Module):
    def __init__(
            self,
            model_name: str,
            sampling_rate,
            feature_extractor: str = None,
            *args, **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)

        self.sampling_rate = sampling_rate

        # TODO: find a way to not use the ft version
        self.model = HubertModel.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            feature_extractor if feature_extractor is not None else model_name
        )
        self.output_dim = self.model.config.hidden_size

    def forward(self, raw_speech: Tensor) -> Tensor:
        # TODO: LLaVA showed that not using the last hidden state
        # but the middle one can improve decoder's accuracy (investigate)
        return self.model(raw_speech).last_hidden_state


if __name__ == "__main__":
    import torch

    dummy_input = torch.randn((8, 258560))

    encoder = Encoder(
        model_name="facebook/hubert-large-ls960-ft",
        feature_extractor="facebook/hubert-base-ls960",
        sampling_rate=16_000,
    )

    output = encoder(raw_speech=dummy_input)

    print(output.shape)
