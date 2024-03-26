import json
import torch

from torch import nn, Tensor
from Encoder import Encoder
from omegaconf import OmegaConf
from dataclasses import dataclass
from collections import OrderedDict
from safetensors import safe_open
from Decoder import Decoder, DecoderInput
from transformers import Wav2Vec2Processor
from LinearProjector import LinearProjector
from FramesDownSampler import FramesDownSampler
from peft import LoraConfig, TaskType, get_peft_model
from typing import Optional, List, Union, Dict, Tuple, Any
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


class SLAM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:

        if "encoder" not in kwargs:
            raise ValueError("SLAM expects to receive a dict named 'encoder' as input. See SLAM.help()")

        if "decoder" not in kwargs:
            raise ValueError("SLAM expects to receive a dict named 'decoder' as input. See SLAM.help()")

        self.cfg = {
            "encoder": OmegaConf.to_container(kwargs.pop("encoder")),
            "decoder": OmegaConf.to_container(kwargs.pop("decoder")),
        }

        super().__init__(*args, **kwargs)

        self._train_projector_only = False

        self.encoder = Encoder(**self.cfg["encoder"])

        self.down_sampler = FramesDownSampler(k=5)

        self.decoder = Decoder(**self.cfg["decoder"])

        self.linear_projector = LinearProjector(
            input_dim=self.encoder.output_dim,
            output_dim=self.decoder.model.config.hidden_size,
        )

        self.processor = Wav2Vec2Processor(
            feature_extractor=self.encoder.feature_extractor,
            tokenizer=self.decoder.tokenizer,
        )

        self._freeze_nonlinear_layers()

    def _freeze_nonlinear_layers(self):
        for name, param in self.named_parameters():
            if not any([linear_indicator in name for linear_indicator in ('fc', 'dense', 'linear')]):
                param.requires_grad = False

    @classmethod
    def help(cls):
        print("TODO: help message")

    @classmethod
    def _load_safetensors(cls, path_to_safetensors) -> Tuple[Dict[str, Any], Dict[str, str]]:
        tensors = {}

        with safe_open(path_to_safetensors, framework="pt") as f:
            metadata = f.metadata()

            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        return tensors, metadata

    @classmethod
    def from_pretrained(
        cls,
        path_to_safetensors: str,
        cfg: Optional[Dict[str, str]] = None,
    ):
        tensors, metadata = cls._load_safetensors(path_to_safetensors)

        cfg = cfg if cfg is not None else metadata.get("cfg")

        if cfg is None:
            raise RuntimeError(
                "No configuration found in the metadata of the provided safetensors file."
                " Please ensure that the safetensors file contains the necessary configuration information"
                " or provide it explicitly through the 'cfg' argument when calling 'from_pretrained'."
            )

        cfg = OmegaConf.create(json.loads(cfg))

        model = SLAM(**cfg)

        if cfg.decoder.get("peft"):

            peft_config = LoraConfig(**{
                "task_type": TaskType.CAUSAL_LM,
                "inference_mode": False,
                **cfg.decoder["peft"],
            })

            model.decoder.model = get_peft_model(
                model=model.decoder.model,
                peft_config=peft_config,
            )

        model.load_state_dict(state_dict=tensors, strict=True)

        return model

    def train(self, mode: bool = True):
        super().train(mode)

        self.encoder.eval()

        if self._train_projector_only is True:
            self.decoder.eval()

        return self

    def eval(self):
        return super().eval()

    def _encode_audio(self, raw_audio: Tensor) -> Tensor:
        audio_embeddings = self.encoder(raw_audio.unsqueeze(0))

        down_sampled_audio_embeddings = self.down_sampler(audio_embeddings)

        return self.linear_projector(down_sampled_audio_embeddings)[0]

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

            if isinstance(_input, dict) and not isinstance(_input, SLAMInput):
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

    # TODO: handle batch as input
    # TODO: change default max_length to decoder's max_length
    # TODO: replace by a SLAM.generate that supports {audio} key word
    # ie. generate(prompt="Transcribe speech to text {audio}", raw_speech)
    def generate_transcript(self, raw_speech: Union[Tensor, List[Tensor]], max_length: int = 512) -> Tensor:
        self._init_processor()

        encoder_input = raw_speech

        # create batch size of one if a single sample is provided
        if isinstance(encoder_input, Tensor) and len(encoder_input.shape) == 1:
            encoder_input = encoder_input.unsqueeze(0)

        device_to_use = next(self.parameters()).device

        decoder_inputs = [DecoderInput(
            instruct=self.decoder.instruct_template.format(instruct="Transcribe speech to text {audio}"),
            audio_embedding=self._encode_audio(_encoder_input),
        ) for _encoder_input in encoder_input]

        # used to normalize the logits (useful for beam search)
        logits_processor = LogitsProcessorList([LogitNormalization()])

        # used to stop inference when max_length is reached
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(len(encoder_input), dtype=torch.long, device=device_to_use)

        eos_token_id_tensor = torch.tensor([self.decoder.tokenizer.eos_token_id], device=device_to_use)

        input_ids = torch.empty((len(encoder_input), 0), dtype=torch.long, device=device_to_use)

        while True:

            outputs: CausalLMOutputWithPast = self.decoder(decoder_inputs, apply_prompt_formating=False)

            next_token_logits = outputs.logits[0, -1, :]

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
            if all(stopping_criteria(input_ids=input_ids, scores=None).cpu().detach().tolist()):
                break

            decoder_inputs = [DecoderInput(
                instruct=_decoder_inputs.instruct + self.decoder.tokenizer.decode(_next_tokens),
                audio_embedding=_decoder_inputs.audio_embedding,
            ) for (_decoder_inputs, _next_tokens) in zip(decoder_inputs, next_tokens)]

        return input_ids


if __name__ == "__main__":
    import torch

    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = SLAM(
        decoder={
            "model_name": "abacaj/phi-2-super"
        },
        encoder={
            "model_name": "facebook/hubert-large-ls960-ft",
            "sampling_rate": 16_000,
        }
    ).eval().to(device_to_use)

    dummy_input_values = SLAMInput(
        instruct=(f"{model.decoder.tokenizer.eos_token}[INST]" " Transcribe speech to text {audio} [/INST]"),
        raw_audio=torch.randn((258560), device=device_to_use),
    )

    model.forward(dummy_input_values)
