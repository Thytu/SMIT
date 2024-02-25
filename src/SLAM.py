from torch import nn, Tensor
from LinearProjector import LinearProjector
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
class Decoder(nn.Module):
    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.prompt_template = "USER:{speech_embeddings}Transcribe speech to text ASSISTANT:"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, speech_embeddings):

        (
            tokens_before_speech_embeddings,
            tokens_after_speech_embeddings,
        ) = self.prompt_template.split("{speech_embeddings}")

        # tokenizing prompt
        tokens_before_speech_embeddings = self.tokenizer(
            tokens_before_speech_embeddings,
            return_tensors="pt"
        ).input_ids
        tokens_after_speech_embeddings = self.tokenizer(
            tokens_after_speech_embeddings,
            return_tensors="pt"
        ).input_ids

        # droping EOS token
        tokens_before_speech_embeddings = tokens_before_speech_embeddings[::, :-1]
        tokens_after_speech_embeddings = tokens_after_speech_embeddings[::, :-1]

        # generating prompt embeddings
        prompt_embeddings_before_speech_embeddings = self.model.get_input_embeddings()(tokens_before_speech_embeddings)
        prompt_embeddings_after_speech_embeddings = self.model.get_input_embeddings()(tokens_after_speech_embeddings)

        # concatenating prompt_embeddings and speech_embeddings into a single tensor
        inputs_embeds = torch.cat(
            (
                prompt_embeddings_before_speech_embeddings,
                speech_embeddings,
                prompt_embeddings_after_speech_embeddings,
            ),
            dim=1,
        )

        # generate next token
        return self.model(inputs_embeds=inputs_embeds).logits


class SLAM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = Encoder(sampling_rate=16_000)

        self.decoder = Decoder(model_name="microsoft/phi-2")

        self.linear_projector = LinearProjector(
            input_dim=self.encoder.output_dim,
            output_dim=self.decoder.model.config.hidden_size,
        )

    # TODO: it must accept batches!
    # currently it takes as an input a single array representing a single audio sample
    def forward(self, raw_speech: Tensor) -> Tensor:

        speech_embeddings = self.encoder(raw_speech)
        speech_embeddings = self.linear_projector(speech_embeddings)

        logits = self.decoder(speech_embeddings=speech_embeddings)

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
