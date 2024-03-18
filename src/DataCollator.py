from torch import Tensor, tensor
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
        return {
            "inputs": [
                {
                    "instruct": feature["inputs"]["instruct"],
                    "raw_audio": tensor(feature["inputs"]["raw_audio"]) if feature["inputs"]["raw_audio"] is not None else None,
                } for feature in features
            ],
            "labels": [tensor(feature["labels"]) for feature in features],
        }
