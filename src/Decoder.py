import torch

from torch import nn
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
