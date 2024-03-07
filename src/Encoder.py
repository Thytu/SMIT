from torch import nn, Tensor
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class Encoder(nn.Module):


    def __init__(self, sampling_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sampling_rate = sampling_rate

        # TODO: find a way to not use the ft version
        self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        self.output_dim = self.model.config.hidden_size

    # TODO: add call to self.feature_extractor when using SLAM.generate_transcript
    def forward(self, raw_speech: Tensor) -> Tensor:
        return self.model(raw_speech).last_hidden_state


if __name__ == "__main__":
    import torch

    dummy_input = torch.randn((8, 258560))
    encoder = Encoder(sampling_rate=16_000)

    output = encoder(raw_speech=dummy_input)

    print(output.shape)
