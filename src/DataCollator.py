from torch import Tensor
from dataclasses import dataclass
from transformers import Wav2Vec2Processor
from typing import Optional, Union, List, Dict


@dataclass
class DataCollator:

    processor: Wav2Vec2Processor

    padding_inputs: Union[bool, str] = True
    max_length_inputs: Optional[int] = None

    padding_labels: Union[bool, str] = True
    max_length_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], Tensor]]]) -> Dict[str, Tensor]:

        # split inputs and labels since they have to be of different lenghts and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding_inputs,
            max_length=self.max_length_inputs,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=True,
            # padding=self.padding_labels,
            # max_length=self.max_length_labels,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
