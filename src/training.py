import wandb
import functools
import numpy as np

from SLAM import SLAM
from typing import Dict
from datasets import load_from_disk
from evaluate import load as load_metric
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

    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer: float = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main():

    model = SLAM(decode_name="abacaj/phi-2-super")

    for name, param in model.named_parameters():
        if not any([linear_indicator in name for linear_indicator in ('fc', 'dense', 'linear')]):
            param.requires_grad = False

    processor = Wav2Vec2Processor(
        feature_extractor=model.encoder.feature_extractor,
        tokenizer=model.decoder.tokenizer,
    )

    data_collator = DataCollator(
        processor=processor,
        padding_inputs=True,
        padding_labels='max_length',
        max_length_labels=512,
    )

    dataset = load_from_disk("outputs/dataset/")

    train_set = dataset['train']
    test_set = dataset['test']
    validation_set = dataset['validation']

    training_args = TrainingArguments(
        output_dir="/scratch/SLAM-ASR-outputs/model/",
        # group_by_length=True, # Makes the training init suepr long (~2h)
        bf16=True,
        per_device_train_batch_size=1,
        # per_device_train_batch_size=4, # OOM
        per_device_eval_batch_size=6,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        logging_steps=100,
        learning_rate=1e-4,
        max_steps=100_000,
        warmup_steps=1_000,
        save_total_limit=15,
        dataloader_num_workers=16,
        report_to="wandb",
        weight_decay=0,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        # compute_metrics=functools.partial(compute_metrics, processor=processor), # creates huge spikes in VRAM uage
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    with wandb.init(project="SLAM-ASR") as run:
        trainer.train()
        # trainer.train(resume_from_checkpoint="/scratch/SLAM-ASR-outputs/model/checkpoint-7000/")
        # trainer.evaluate(
        #     eval_dataset=validation_set,
        #     metric_key_prefix="validation"
        # )


if __name__ == "__main__":
    main()
