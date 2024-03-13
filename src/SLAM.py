import torch

from torch import nn, Tensor
from Encoder import Encoder
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, List, Union, Dict
from Decoder import Decoder, DecoderInput
from transformers import Wav2Vec2Processor
from LinearProjector import LinearProjector
from FramesDownSampler import FramesDownSampler
from huggingface_hub import PyTorchModelHubMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import (
    LogitNormalization,
    LogitsProcessorList,
    MaxLengthCriteria,
    StoppingCriteriaList,
)


@dataclass
class SLAMInput(OrderedDict):
    instruct: str = None
    instruct_ids: Optional[Union[List[int], Tensor]] = None
    raw_audio: Optional[Tensor] = None
    # labels: Optional[str] = None



class SLAM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, decode_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = Encoder(sampling_rate=16_000)
        self.down_sampler = FramesDownSampler(k=5)
        self.decoder = Decoder(model_name=decode_name, **kwargs)
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

        # Only train decoder and linear_projector
        self.encoder.eval()

        # Only train Linear layers
        for name, param in self.decoder.named_parameters():
            if not any([linear_indicator in name for linear_indicator in ('fc', 'dense', 'linear')]):
                param.requires_grad = False

        return self

    def eval(self):
        return super().eval()

    # def _prepare_inputs_for_decoder(self, input_values: Tensor) -> Tensor:
    #     speech_embeddings = self.encoder(input_values)

    #     down_sampled_speech_embeddings = self.down_sampler(speech_embeddings)

    #     return self.linear_projector(down_sampled_speech_embeddings)

    def _encode_audio(self, raw_audio: Tensor) -> Tensor:
        audio_embeddings = self.encoder(raw_audio.unsqueeze(0))

        down_sampled_audio_embeddings = self.down_sampler(audio_embeddings)

        return self.linear_projector(down_sampled_audio_embeddings)[0]

    # TODO LIST
    # SLAM.forward must accept non-audio input
    # Add text instruct dataset to dataset

    def forward(
        self,
        inputs: Union[
            SLAMInput,
            Dict[str, Union[str, Tensor, None]],
            List[SLAMInput],
            List[Dict[str, Union[str, Tensor, None]]],
        ],
        labels: Optional[Tensor] = None,
    ) -> CausalLMOutputWithPast:

        if isinstance(inputs, (dict, SLAMInput)):
            inputs = [inputs]

        for idx, _input in enumerate(inputs):

            if isinstance(_input, dict):
                inputs[idx] = SLAMInput(
                    instruct=_input.get("instruct"),
                    instruct_ids=_input.get("instruct_ids"),
                    raw_audio=_input["raw_audio"],
                )

            if inputs[idx].instruct is None and inputs[idx].instruct_ids is None:
                raise RuntimeError("SLAMInput must contains either instruct or instruct_ids.")

        if labels is None:
            labels = [None] * len(inputs)

        inputs = [DecoderInput(
            instruct=_input.instruct,
            instruct_ids=_input.instruct_ids,
            audio_embedding=self._encode_audio(_input.raw_audio) if _input.raw_audio is not None else None,
            labels=_labels,
        ) for (_input, _labels) in zip(inputs, labels)]

        return self.decoder(inputs)

    # TODO: change default max_length to decoder's max_length
    # TODO: change function to generic .generate one but it must handles audio as input
    def generate_transcript(self, raw_speech: Tensor, max_length: int = 512) -> Tensor:
        self._init_processor()

        encoder_input = raw_speech

        # create batch size of one if a single sample is provided
        if len(encoder_input.shape) == 1:
            encoder_input = encoder_input.unsqueeze(0)

        device_to_use = next(self.parameters()).device

        speech_embeddings = self._prepare_inputs_for_decoder(encoder_input)

        inputs_embedding = self.decoder._generate_inputs_embeds(speech_embeddings)

        # used to normalize the logits (useful for beam search)
        logits_processor = LogitsProcessorList([LogitNormalization()])

        # used to stop inference when max_length is reached
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(encoder_input.shape[0], dtype=torch.long, device=device_to_use)

        eos_token_id_tensor = torch.tensor([self.decoder.tokenizer.eos_token_id], device=device_to_use)

        input_ids = torch.empty((encoder_input.size(0), 0), dtype=torch.long, device=device_to_use)

        while True:

            outputs: CausalLMOutputWithPast = self.decoder(inputs_embedding=inputs_embedding)

            next_token_logits = outputs.logits[:, -1, :]

            next_tokens_scores = logits_processor(input_ids=None, scores=next_token_logits)

            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + self.decoder.tokenizer.pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # update which sequences are finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids=input_ids, scores=None):
                break

            new_inputs_embedding: Tensor = self.decoder.model.get_input_embeddings()(next_tokens)

            inputs_embedding.inputs_embeds = torch.cat(
                (
                    inputs_embedding.inputs_embeds,
                    new_inputs_embedding.unsqueeze(1),
                ),
                dim=1,
            )

        if len(raw_speech.shape) == 1:
            input_ids = input_ids[0]

        return input_ids


if __name__ == "__main__":
    import torch

    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = SLAM(decode_name="abacaj/phi-2-super").eval().to(device_to_use)

    # dummy_input_values = torch.randn((2, 258560), device=device_to_use)

    dummy_input_values = SLAMInput(
        instruct=(f"{model.decoder.tokenizer.eos_token}[INST]" " Transcribe speech to text {audio} [/INST]"),
        raw_audio=torch.randn((258560), device=device_to_use),
    )

    model.forward(dummy_input_values)

    # FIXME
    # input_ids = model.generate_transcript(dummy_input_values)

    # print(f"{input_ids.shape=}")

    # for _input_ids in input_ids:
    #     print("[TRANSCRIPTION]\n-----\n", model.decoder.tokenizer.decode(_input_ids), end="\n-----\n\n")
