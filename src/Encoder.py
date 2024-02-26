from torch import nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class Encoder(nn.Module):
    def __init__(self, sampling_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sampling_rate = sampling_rate
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

        # TODO: find a way to not use the ft version
        self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        self.output_dim = self.model.config.hidden_size

    def forward(self, raw_speech):

        features = self.feature_extractor(
            raw_speech=raw_speech,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        return self.model(features.input_values).last_hidden_state
