from torch import Tensor, tensor
from dataclasses import dataclass
from transformers import Wav2Vec2Processor
from typing import Optional, Union, List, Dict


from SLAM import SLAMInput

@dataclass
class DataCollator:

    processor: Wav2Vec2Processor

    padding_inputs: Union[bool, str] = True
    max_length_inputs: Optional[int] = None

    padding_labels: Union[bool, str] = True
    max_length_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], Tensor]]]) -> Dict[str, Tensor]:
        # return features

        # return {
        output = {
            "inputs": [
                {
                    "instruct": feature["inputs"]["instruct"],
                    "raw_audio": tensor(feature["inputs"]["raw_audio"]) if feature["inputs"]["raw_audio"] is not None else None,
                } for feature in features
            ],
            # ],
            "labels": [tensor(feature["labels"]) for feature in features],
        }
        return output

        # split inputs and labels since they have to be of different lenghts and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # SLAMInput(
        #     prompt=(f"{processor.tokenizer.eos_token}[INST]" " Transcribe speech to text {audio} [/INST]"),
        #     raw_audio=value,
        # )

        # batch = self.processor.pad(
        #     input_features,
        #     padding=self.padding_inputs,
        #     max_length=self.max_length_inputs,
        #     return_tensors="pt",
        # )

        # labels_batch = self.processor.pad(
        #     labels=label_features,
        #     padding=True,
        #     # padding=self.padding_labels,
        #     # max_length=self.max_length_labels,
        #     return_tensors="pt",
        # )

        # labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
