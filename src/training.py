import wandb
import hydra
import numpy as np

from SLAM import SLAM
from typing import Dict
from datasets import load_from_disk
from evaluate import load as load_metric
from omegaconf import DictConfig
from DataCollator import DataCollator
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    EarlyStoppingCallback,
)


def compute_metrics(pred, processor: Wav2Vec2Processor) -> Dict[str, float]:

    wer_metric = load_metric("wer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer: float = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@hydra.main(version_base=None, config_path="../conf", config_name="default")
def main(cfg : DictConfig):

    model = SLAM(**cfg.model)

    processor = Wav2Vec2Processor(
        feature_extractor=model.encoder.feature_extractor,
        tokenizer=model.decoder.tokenizer,
    )

    data_collator = DataCollator(
        processor=processor,
        padding_inputs=True,
        padding_labels='max_length',
        max_length_labels=model.decoder.tokenizer.model_max_length,
    )

    dataset = load_from_disk("outputs/dataset/")

    train_set = dataset['train']
    test_set = dataset['test']
    validation_set = dataset['validation']

    training_args = TrainingArguments(**cfg.training_args)

    callbacks = []
    if (early_stopping_patience := cfg.training.early_stopping_patience) is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        # compute_metrics=functools.partial(compute_metrics, processor=processor), # creates huge spikes in VRAM uage
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )

    with wandb.init(project="SLAM-ASR") as run:
        trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
        trainer.evaluate(
            eval_dataset=validation_set,
            metric_key_prefix="validation"
        )


if __name__ == "__main__":
    main()
