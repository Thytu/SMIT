import torch

from typing import Dict
from SLAM import SLAMInput
from transformers import Wav2Vec2Processor
from datasets import Dataset, load_dataset, Audio, concatenate_datasets

LIBRISPEECH_ASR = "librispeech_asr"
SUBSET = "all"
SAMPLING_RATE = 16_000


def load_processed_librispeech_dataset(
    train_split_size: str = None,
    test_split_size: str = None,
    validation_split_size: str = None,
) -> Dict[str, Dataset]:

    COLUMNS_TO_DROP = ["file", "speaker_id", "chapter_id", "id"]

    common_voice_train = load_dataset(LIBRISPEECH_ASR, "all", split="train.other.500" + (train_split_size if train_split_size is not None else ""))
    common_voice_test = load_dataset(LIBRISPEECH_ASR, "all", split="test.other" + (test_split_size if test_split_size is not None else ""))
    common_voice_validation = load_dataset(LIBRISPEECH_ASR, "all", split="test.other" + (validation_split_size if validation_split_size is not None else ""))

    # TODO: I don't think I need to do that as I can directly use sample["audio"]["array"]
    common_voice_train = common_voice_train.remove_columns(COLUMNS_TO_DROP)
    common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    common_voice_test = common_voice_test.remove_columns(COLUMNS_TO_DROP)
    common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    common_voice_validation = common_voice_validation.remove_columns(COLUMNS_TO_DROP)
    common_voice_validation = common_voice_validation.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    return {
        "train": common_voice_train.shuffle(),
        "test": common_voice_test.shuffle(),
        "validation": common_voice_validation.shuffle(),
    }


def load_processed_instruct_dataset(
    test_size: float,
) -> Dict[str, Dataset]:

    instruct_dataset = load_dataset(
        "WizardLM/WizardLM_evol_instruct_70k",
        split="train",
    ).train_test_split(test_size=test_size)

    return {
        "train": instruct_dataset["train"].shuffle(),
        "test": instruct_dataset["test"].shuffle(),
    }


def add_raw_speech_feature_to_librispeech_dataset(batch, processor: Wav2Vec2Processor):

    resampled_audio = processor(
        batch["audio"]["array"],
        sampling_rate=batch["audio"]["sampling_rate"]
    ).input_values[0]

    batch["inputs"] = {
        "instruct": "Transcribe speech to text {audio}",
        "raw_audio": resampled_audio,
    }
    batch["input_length"] = len(batch["inputs"])

    batch["labels"] = processor(
        text=batch["text"].capitalize() + ".",
    ).input_ids

    if batch["labels"][-1] != processor.tokenizer.eos_token_id:
        batch["labels"].append(processor.tokenizer.eos_token_id)

    return batch


def process_instruct_dataset(batch):

    batch["inputs"] = {
        "instruct": batch["instruction"],
        "raw_audio": None,
    }
    batch["input_length"] = len(batch["inputs"])

    batch["labels"] = processor(text=batch["output"]).input_ids

    # TODO: Verify if I should add the EOS token
    if len(batch["labels"]) > 0 and batch["labels"][-1] != processor.tokenizer.eos_token_id:
        batch["labels"].append(processor.tokenizer.eos_token_id)

    return batch



if __name__ == "__main__":

    import os
    import random

    from Encoder import Encoder
    from Decoder import Decoder
    from datasets import DatasetDict

    MAX_LENGTH = 2048
    SAFETY_MARGIN = 350 # for the prompt template
    # FIXME: I shouldn't need such a huge sage margin

    _dataset = load_processed_librispeech_dataset(
        # train_split_size="[:50]",
        # test_split_size="[:50]",
        # validation_split_size="[:50]",
    )

    instruct_dataset = load_processed_instruct_dataset(test_size=0.2)
    instruct_dataset["validation"] = instruct_dataset["test"]

    rand_int = random.randint(0, len(_dataset['train'])-1)

    print("Target text:", _dataset['train'][rand_int]["text"])
    print("Input array shape:", _dataset['train'][rand_int]["audio"]["array"].shape)
    print("Sampling rate:", _dataset['train'][rand_int]["audio"]["sampling_rate"])

    processor = Wav2Vec2Processor(
        feature_extractor=Encoder(sampling_rate=SAMPLING_RATE).feature_extractor,
        tokenizer=Decoder(model_name="microsoft/phi-2").tokenizer,
    )

    os.makedirs("outputs/dataset/", exist_ok=True)

    for set_name in ("train", "test", "validation"):

        nb_samples_from_each_dataset = min(len(instruct_dataset[set_name]), len(_dataset[set_name]))
        instruct_dataset[set_name] = instruct_dataset[set_name].select([i for i in range(nb_samples_from_each_dataset)])

        instruct_dataset[set_name] = instruct_dataset[set_name].map(
            process_instruct_dataset,
            remove_columns=instruct_dataset[set_name].column_names,
            num_proc=max(1, os.cpu_count() - 1),
        ).filter(lambda sample: len(sample["labels"]) > 0 and len(sample["inputs"]["instruct"]) + len(sample["labels"]) < MAX_LENGTH - SAFETY_MARGIN)

        # Update after filtering instruct dataset
        nb_samples_from_each_dataset = min(len(instruct_dataset[set_name]), len(_dataset[set_name]))
        _dataset[set_name] = _dataset[set_name].select([i for i in range(nb_samples_from_each_dataset)])

        _dataset[set_name] = _dataset[set_name].map(
            add_raw_speech_feature_to_librispeech_dataset,
            remove_columns=_dataset[set_name].column_names,
            fn_kwargs={"processor": processor},
            num_proc=max(1, os.cpu_count() - 1),
        )

        _dataset[set_name] = concatenate_datasets([
            _dataset[set_name],
            instruct_dataset[set_name]
        ]).shuffle()

    DatasetDict(_dataset).save_to_disk(
        dataset_dict_path=f"outputs/dataset/",
        num_proc=max(1, os.cpu_count() - 1),
    )
