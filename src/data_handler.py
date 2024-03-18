import os
import hydra
import fn_factory

from Encoder import Encoder
from Decoder import Decoder
from omegaconf import DictConfig
from transformers import Wav2Vec2Processor
from datasets import DatasetDict, load_dataset, Audio, concatenate_datasets


def preprocess_samples(
    batch,
    audio_column: str,
    label_column: str,
    instruct_column: str,
    instruct: str,
    processor: Wav2Vec2Processor,
):

    raw_audio = processor(
        batch[audio_column]["array"],
        sampling_rate=batch[audio_column]["sampling_rate"]
    ).input_values[0] if audio_column is not None else None

    batch["inputs"] = {
        "instruct": instruct if instruct is not None else batch[instruct_column],
        "raw_audio": raw_audio,
    }
    batch["input_length"] = len(batch["inputs"])

    batch["labels"] = processor(
        text=batch[label_column].capitalize() + ".", # TODO: handle label preprocessing
    ).input_ids

    if batch["labels"][-1] != processor.tokenizer.eos_token_id:
        batch["labels"].append(processor.tokenizer.eos_token_id)

    return batch


@hydra.main(version_base=None, config_path="../conf", config_name="default")
def main(cfg : DictConfig) -> DatasetDict:

    splits = {
        "train": [],
        "test": [],
        "validation": [],
    }

    processor = Wav2Vec2Processor(
        feature_extractor=Encoder(**cfg.model.encoder).feature_extractor,
        tokenizer=Decoder(model_name=cfg.model.decoder.model_name).tokenizer,
    )

    for _dataset_name, _dataset_cfg in cfg.datasets.items():
        for split in ("train", "test", "validation"):

            _samples = load_dataset(
                _dataset_name,
                name=_dataset_cfg.get("subset"),
                split=_dataset_cfg.get("splits", {}).get(split),
            )

            if (audio_column := _dataset_cfg.get("audio_column")) is not None:
                _samples = _samples.cast_column(audio_column, Audio(sampling_rate=cfg.model.encoder.sampling_rate))

            _samples = _samples.map(
                preprocess_samples,
                fn_kwargs={
                    "audio_column": audio_column,
                    "label_column": _dataset_cfg.get("label_column"),
                    "instruct_column": _dataset_cfg.get("instruct_column"),
                    "instruct": _dataset_cfg.get("instruct"),
                    "processor": processor,
                },
                remove_columns=_samples.column_names,
                num_proc=max(1, os.cpu_count() - 1),
            )

            for filter_to_apply, fn_args in _dataset_cfg.get("filters", {}).items():

                # TODO: instead of applying iteratively, I could concate the bools
                # i.e filter(lambda sample: all([fn(element) for fn in filters]))

                _samples = getattr(fn_factory, filter_to_apply)(
                    **fn_args,
                    dataset=_samples,
                    tokenizer=processor.tokenizer
                )

            splits[split].append(_samples)

    os.makedirs("outputs/dataset/", exist_ok=True)

    DatasetDict({
        k: concatenate_datasets(v).shuffle() for (k, v) in splits.items()
    }).save_to_disk(
        dataset_dict_path=f"outputs/dataset/",
        num_proc=max(1, os.cpu_count() - 1),
    )


if __name__ == "__main__":
    main()
