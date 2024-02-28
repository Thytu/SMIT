import torch

from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
from transformers import Wav2Vec2Processor
from datasets import Dataset, load_dataset, Audio


LIBRISPEECH_ASR = "librispeech_asr"
SUBSET = "all"
SAMPLING_RATE = 16_000


def downsample_frames(frames: torch.Tensor, k: int = 5):
    """Down sample frames by concatenating every k consecutive frames in the feature dimension.

    Args:
        frames (torch.Tensor): frames to concatenate, usually encoder ouput
        k (int, optional): Downsampling rate. Defaults to 5.

    Returns:
        torch.Tensor: concatenated frames
    """

    frames = frames[::,:frames.size(1) - frames.size(1) % k]

    chunks = torch.chunk(frames, frames.size(1) // k, dim=1)

    concatenated_frames = torch.stack(chunks).permute(1, 0, 2, 3)

    downsampled_frames = torch.sum(concatenated_frames, dim=2)

    return downsampled_frames


def load_processed_dataset(
    train_split_size: str = None,
    test_split_size: str = None,
    validation_split_size: str = None,
) -> Dict[str, Dataset]:

    COLUMNS_TO_DROP = ["file", "speaker_id", "chapter_id", "id"]

    common_voice_train = load_dataset(LIBRISPEECH_ASR, "all", split="train.other.500" + (train_split_size if train_split_size is not None else ""))
    common_voice_test = load_dataset(LIBRISPEECH_ASR, "all", split="test.other" + (test_split_size if test_split_size is not None else ""))
    common_voice_validation = load_dataset(LIBRISPEECH_ASR, "all", split="test.other" + (validation_split_size if validation_split_size is not None else ""))

    # TODO: I don't think I need to do that as I can directly use sample["audio"]["array"]
    for _dataset in (common_voice_train, common_voice_test, common_voice_validation):
        _dataset = _dataset.remove_columns(COLUMNS_TO_DROP)
        _dataset = _dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    return {
        "train": common_voice_train.shuffle(),
        "test": common_voice_test.shuffle(),
        "validation": common_voice_validation.shuffle(),
    }


def add_raw_speech_feature_to_dataset(batch, processor):

    audio = batch["audio"]

    batch["input_features"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    batch["input_length"] = len(batch["input_features"])

    batch["labels"] = processor(text=batch["text"]).input_ids

    return batch


@dataclass
class DataCollator:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        # split inputs and labels since they have to be of different lenghts and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        # TODO: verify the -100, I'm totally not sure about it
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch



# from datasets import load_dataset

# model = SLAM()

# librispeech_train = load_dataset("librispeech_asr", "all", split="train.clean.100", trust_remote_code=True)

# sample = next(iter(librispeech_train))["audio"]["array"]

# output = model.generate_transcript(sample)

# print(output)



if __name__ == "__main__":

    import random

    from Encoder import Encoder
    from Decoder import Decoder


    _dataset = load_processed_dataset(
        train_split_size="[:1%]",
        test_split_size="[:10%]",
        validation_split_size="[:10%]",
    )

    rand_int = random.randint(0, len(_dataset['train'])-1)

    print("Target text:", _dataset['train'][rand_int]["text"])
    print("Input array shape:", _dataset['train'][rand_int]["audio"]["array"].shape)
    print("Sampling rate:", _dataset['train'][rand_int]["audio"]["sampling_rate"])

    processor = Wav2Vec2Processor(
        feature_extractor=Encoder(sampling_rate=SAMPLING_RATE).feature_extractor,
        tokenizer=Decoder(model_name="microsoft/phi-2").tokenizer,
    )

    for set_name in ("train", "test", "validation"):
        _dataset[set_name] = _dataset[set_name].map(
            add_raw_speech_feature_to_dataset,
            remove_columns=_dataset[set_name].column_names,
            fn_kwargs={"processor": processor},
        )

    print("After mapping:", _dataset['train'][rand_int].keys())
