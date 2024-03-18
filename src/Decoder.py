import torch

from torch import nn, Tensor
from logging import getLogger
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class DecoderInput(OrderedDict):
    instruct: Optional[str] = None
    instruct_ids: Optional[Union[List[int], Tensor]] = None
    audio_embedding: Optional[Tensor] = None
    labels: Optional[str] = None


class Decoder(nn.Module):
    def __init__(
            self,
            model_name: str,
            audio_placeholder: Optional[str] = None,
            prompt_template: Optional[str] = None,
            *args, **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)

        self.logger = getLogger(__name__)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.audio_placeholder: str = audio_placeholder if audio_placeholder is not None else "{audio}"

        # TODO: provide guielines on how to provide a prompt template + check that the prompt template is valid
        default_prompt_template = (f"{self.tokenizer.eos_token}[INST]" " {instruct} [/INST]")
        self.instruct_template = prompt_template if prompt_template is not None else default_prompt_template

        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)

    def _generate_embedding_without_audio(self, prompt: str, device_to_use: str):

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids[0].to(device_to_use)

        if input_ids[-1] == self.tokenizer.eos_token_id:
            input_ids = input_ids[:-1]

        embeddings = self.model.get_input_embeddings()(input_ids)

        return embeddings

    def _generate_embedding_with_audio(self, prompt: str, audio_embedding: Tensor, device_to_use: str):

        tokens_before_speech_embeddings = self.tokenizer(
            prompt.split(self.audio_placeholder)[0],
            return_tensors="pt"
        ).input_ids.to(device_to_use)[0]

        tokens_after_speech_embeddings = self.tokenizer(
            prompt.split(self.audio_placeholder)[1],
            return_tensors="pt"
        ).input_ids.to(device_to_use)[0]

        # droping EOS token
        if tokens_before_speech_embeddings[-1] == self.tokenizer.eos_token_id:
            tokens_before_speech_embeddings = tokens_before_speech_embeddings[:-1]

        if tokens_after_speech_embeddings[-1] == self.tokenizer.eos_token_id:
            tokens_after_speech_embeddings = tokens_after_speech_embeddings[:-1]

        prompt_embeddings_before_speech_embeddings: Tensor = self.model.get_input_embeddings()(tokens_before_speech_embeddings)
        prompt_embeddings_after_speech_embeddings: Tensor = self.model.get_input_embeddings()(tokens_after_speech_embeddings)

        # concatenating prompt and audio_embedding into a single tensor
        embeddings = torch.cat(
            (
                prompt_embeddings_before_speech_embeddings,
                audio_embedding,
                prompt_embeddings_after_speech_embeddings,
            ),
        )

        return embeddings

    def forward(
        self,
        inputs: List[DecoderInput],
        apply_prompt_formating: bool = True,
    ) -> CausalLMOutputWithPast:

        device_to_use = next(self.parameters()).device

        if isinstance(inputs, DecoderInput):
            inputs = [inputs]

        # casting to DecoderInput.instruct_ids
        for _input in inputs:

            if _input.instruct is None and _input.instruct_ids is None:
                raise RuntimeError("DecoderInput must contains either instruct or instruct_ids.")

            if _input.instruct is None:
                _input.instruct = self.tokenizer.decode(_input.instruct_ids)
                _input.instruct_ids = None

        labels = None
        inputs_embeddings = []
        pad_embedding: Tensor = self.model.get_input_embeddings()(torch.tensor(self.tokenizer.pad_token_id, device=device_to_use))

        if any([_input.labels is not None for _input in inputs]):
            if not all([_input.labels is not None for _input in inputs]):
                raise RuntimeError("Received an input with DecoderInput.audio_embedding different that None and one being None.")

            labels = []

        for _input in inputs:

            if self.audio_placeholder in _input.instruct and _input.audio_embedding is None:
                raise RuntimeError(f"Received a prompt with the '{self.audio_placeholder}' key but with DecoderInput.audio_embedding being None")

            if self.audio_placeholder not in _input.instruct and _input.audio_embedding is not None:
                self.logger.warning((
                    f"Received a prompt without an '{self.audio_placeholder}' key but DecoderInput.audio_embedding being not None. "
                    "DecoderInput.audio_embedding will be ignored."
                ))

            # Applying decoder prompt format around the instruct
            if apply_prompt_formating:
                _input.instruct = self.instruct_template.format(instruct=_input.instruct)

            if self.audio_placeholder in _input.instruct:
                embedding = self._generate_embedding_with_audio(
                    prompt=_input.instruct,
                    audio_embedding=_input.audio_embedding,
                    device_to_use=device_to_use,
                )
            else:
                embedding = self._generate_embedding_without_audio(prompt=_input.instruct, device_to_use=device_to_use)

            if self.tokenizer.model_max_length is not None and embedding.size(0) > self.tokenizer.model_max_length:
                raise RuntimeError(f"Received an input that its embedding is larger than {self.tokenizer.model_max_length=}.")

            if _input.labels is not None:

                _labels = _input.labels
                if isinstance(_labels, str):
                    _labels: Tensor = self.tokenizer(_labels, return_tensors="pt").input_ids[0].to(device_to_use)

                if _labels[-1] != self.tokenizer.eos_token_id:
                    _labels = torch.cat([_labels, torch.tensor([self.tokenizer.eos_token_id], device=device_to_use)], dim=-1)

                # adding labels to input
                inputs_to_add = _labels.clone()
                inputs_to_add[inputs_to_add == -100] = self.tokenizer.pad_token_id

                labels_embeddings = self.model.get_input_embeddings()(inputs_to_add)

                # ignoring input when calculating loss
                _labels = torch.cat(
                    (
                        torch.full(
                            size=[embedding.size(0)],
                            fill_value=-100,
                            device=device_to_use,
                        ),
                        _labels.to(device_to_use),
                    ),
                    dim=-1
                )

                embedding = torch.cat(
                    (
                        embedding,
                        labels_embeddings,
                    ),
                    dim=0,
                )

                labels.append(_labels)

            inputs_embeddings.append(embedding)

        max_length = max([embedding.size(0) for embedding in inputs_embeddings])

        inputs_embeddings: Tensor = torch.stack([torch.cat((
            embedding,
            pad_embedding.repeat([max_length - embedding.size(0), 1]),
        )) for embedding in inputs_embeddings])

        if labels is not None:

            # Fills with -100 has we do not calculate loss on pad tokens
            labels = [torch.cat([
                _label,
                torch.full(
                    [max_length - _label.size(0)],
                    fill_value=-100,
                    device=device_to_use
                ),
            ]) for _label in labels]

            labels = torch.stack(labels)

        return self.model(inputs_embeds=inputs_embeddings, labels=labels)


if __name__ == "__main__":
    import torch

    device_to_use = "cpu"

    dummy_input = torch.randn([8, 200, 2560], device=device_to_use)

    decoder = Decoder(model_name="microsoft/phi-2").to(device_to_use)

    dummy_input = DecoderInput(
        instruct=f"Transcribe speech to text {decoder.audio_placeholder}",
        audio_embedding=torch.randn([100, 2560], device=device_to_use),
    )

    output = decoder(dummy_input)

    dummy_input = DecoderInput(
        instruct_ids=decoder.tokenizer(f"Transcribe speech to text {decoder.audio_placeholder}").input_ids,
        audio_embedding=torch.randn([100, 2560], device=device_to_use),
    )

    output = decoder(dummy_input)

    print(f"{output.logits.shape=}")
