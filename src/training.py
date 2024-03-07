import os
import torch
import wandb
import numpy as np
import functools

from SLAM import SLAM
from typing import Dict
from data_handler import DataCollator
from transformers import Wav2Vec2Processor
from datasets import load_dataset, load_metric, load_from_disk
from transformers import TrainingArguments, Trainer


def compute_metrics(pred, processor: Wav2Vec2Processor) -> Dict[str, float]:

    wer_metric = load_metric("wer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer: float = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main():

    model = SLAM()

    for name, param in model.named_parameters():
        if not any([linear_indicator in name for linear_indicator in ('fc', 'dense', 'linear')]):
            param.requires_grad = False

    processor = Wav2Vec2Processor(
        feature_extractor=model.encoder.feature_extractor,
        tokenizer=model.decoder.tokenizer,
    )

    data_collator = DataCollator(
        processor=processor,
        padding=True,
    )

    dataset = load_from_disk("outputs/dataset/")

    train_set = dataset['train']
    test_set = dataset['test']

    # # TODO:
    # # [X] Only train Linear layers
    # # [ ] Only train Decoder
    # # [X] Set batch size as 6
    # # [X] Use Cross Entropy as loss
    # # [X] Use AdamW as optimizer (by default)
    # # [X] max learning rate of 1 × 10−4 without a weight decay
    # # [ ] warmup at the first 1, 000 steps and then keep the maximum learning rate for training all the time.
    # # [ ] max training step is set to 100, 000, but we will stop early if the loss on the validation set does not decrease

    training_args = TrainingArguments(
        output_dir="/scratch/SLAM-ASR-outputs/model/",
        group_by_length=True,
        per_device_train_batch_size=6,
        # gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        # gradient_checkpointing=True,
        # fp16=True,
        # bf16=True,
        save_steps=1000,
        # eval_steps=500,
        eval_steps=5000,
        logging_steps=100,
        learning_rate=1e-4,
        max_steps=100_000,
        warmup_steps=1_000,
        save_total_limit=15,
        dataloader_num_workers=16,
        report_to="wandb",
        weight_decay=0,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=functools.partial(compute_metrics, processor=processor),
        train_dataset=train_set,
        eval_dataset=test_set, # TODO: fix eval loop
        tokenizer=processor.feature_extractor, # TODO: Should I really but processor.feature_extractor here?
    )

    with wandb.init(project="SLAM-ASR") as run:
        trainer.train()


if __name__ == "__main__":
    main()
