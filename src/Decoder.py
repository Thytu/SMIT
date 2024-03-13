import torch

from typing import Optional
from torch import nn, Tensor
from dataclasses import dataclass
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class DecoderInputsEmbedding(OrderedDict):
    inputs_embeds: Tensor
    transcription_embeddings: Optional[Tensor] = None


class Decoder(nn.Module):
    def __init__(self, model_name: str, prompt_template: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        default_prompt_template = (f"{self.tokenizer.eos_token}[INST]" " Transcribe speech to text {speech_embeddings} [/INST]")

        self.prompt_template = prompt_template if prompt_template is not None else default_prompt_template

        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)

    def _generate_inputs_embeds(self, speech_embeddings: Tensor, labels: Optional[torch.Tensor] = None) -> DecoderInputsEmbedding:

        device_to_use = next(self.parameters()).device

        (
            tokens_before_speech_embeddings,
            tokens_after_speech_embeddings,
        ) = self.prompt_template.split("{speech_embeddings}")

        # tokenizing prompt
        tokens_before_speech_embeddings = self.tokenizer(
            tokens_before_speech_embeddings,
            return_tensors="pt"
        ).input_ids.to(device_to_use)
        tokens_after_speech_embeddings = self.tokenizer(
            tokens_after_speech_embeddings,
            return_tensors="pt"
        ).input_ids.to(device_to_use)

        # droping EOS token
        tokens_before_speech_embeddings = tokens_before_speech_embeddings[::, :-1]
        tokens_after_speech_embeddings = tokens_after_speech_embeddings[::, :-1]

        # generating prompt embeddings
        prompt_embeddings_before_speech_embeddings = self.model.get_input_embeddings()(tokens_before_speech_embeddings).repeat((speech_embeddings.size(0), 1, 1))
        prompt_embeddings_after_speech_embeddings = self.model.get_input_embeddings()(tokens_after_speech_embeddings).repeat((speech_embeddings.size(0), 1, 1))

        # concatenating prompt_embeddings and speech_embeddings into a single tensor
        inputs_embeds = torch.cat(
            (
                prompt_embeddings_before_speech_embeddings,
                speech_embeddings,
                prompt_embeddings_after_speech_embeddings,
            ),
            dim=1,
        )

        transcription_embeddings = None

        if labels is not None:

            labels_tokens = labels.clone()
            labels_tokens[labels_tokens == -100] = self.tokenizer.pad_token_id

            transcription_embeddings = self.model.get_input_embeddings()(labels_tokens)

            inputs_embeds = torch.cat(
                (
                    inputs_embeds,
                    transcription_embeddings,
                ),
                dim=1,
            )

        return DecoderInputsEmbedding(
            inputs_embeds=inputs_embeds,
            transcription_embeddings=transcription_embeddings,
        )

    def forward(
        self,
        speech_embeddings: Tensor = None,
        inputs_embedding: DecoderInputsEmbedding = None,
        labels: Optional[torch.Tensor] = None
    ) -> CausalLMOutputWithPast:

        if inputs_embedding is None:
            inputs_embedding = self._generate_inputs_embeds(speech_embeddings, labels)

        if labels is not None:
            size_of_prompt = inputs_embedding.inputs_embeds.size(1) - inputs_embedding.transcription_embeddings.size(1)
            labels = torch.cat(
                (
                    torch.full(
                        size=(inputs_embedding.inputs_embeds.size(0), size_of_prompt),
                        fill_value=-100,
                        device=inputs_embedding.inputs_embeds.device
                    ),
                    labels,
                ),
                dim=1
            )

        return self.model(inputs_embeds=inputs_embedding.inputs_embeds, labels=labels)


if __name__ == "__main__":
    import torch

    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"

    dummy_input = torch.randn([8, 500, 2560], device=device_to_use)

    decoder = Decoder(model_name="microsoft/phi-2").to(device_to_use)

    output = decoder(dummy_input)

    print(f"{output.logits.shape=}")
