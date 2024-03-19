import os
import enum
import wandb
import hydra
import torch
import numpy as np

from SLAM import SLAM
from omegaconf import DictConfig
from typing import Dict, Optional
from contextlib import nullcontext
from datasets import load_from_disk
from evaluate import load as load_metric
from DataCollator import DataCollator
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model


class TrainingStep(str, enum.Enum):
    PRETRAINING = "pretraining"
    TRAINING = "training"



def compute_metrics(pred, processor: Wav2Vec2Processor) -> Dict[str, float]:

    wer_metric = load_metric("wer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer: float = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def train_model(
    step: str,
    cfg : DictConfig,
    path_to_projector: Optional[str] = None,
):

    model = SLAM(**cfg.model)

    if path_to_projector is not None:
        model.linear_projector.load_state_dict(
            state_dict=torch.load(path_to_projector),
            strict=True
        )

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

    if step == TrainingStep.PRETRAINING:
        model._train_projector_only = True
        train_set = train_set.filter(lambda sample : sample["inputs"].get('raw_audio') is not None, num_proc=os.cpu_count() - 1)
        test_set = test_set.filter(lambda sample : sample["inputs"].get('raw_audio') is not None, num_proc=os.cpu_count() - 1)
        validation_set = validation_set.filter(lambda sample : sample["inputs"].get('raw_audio') is not None, num_proc=os.cpu_count() - 1)

    training_args = TrainingArguments(**cfg[step].training_args)

    callbacks = []
    if (early_stopping_patience := cfg[step].early_stopping_patience) is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    if cfg.model.decoder.get("peft"):
        peft_config = LoraConfig(**{
            "task_type": TaskType.CAUSAL_LM,
            "inference_mode": False,
            **cfg.model.decoder["peft"],
        })

        model.decoder.model = get_peft_model(
            model=model.decoder.model,
            peft_config=peft_config,
        )

    # TODO: save only the projector, not encoder/decoder
    # CF https://stackoverflow.com/questions/69651839/is-there-a-way-to-save-only-the-model-with-huggingface-trainer
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

    ctx = wandb.init(project=project) if (project := cfg[step].get("wandb_project_name")) is not None else nullcontext

    with ctx:
        trainer.train(
            resume_from_checkpoint=cfg[step].resume_from_checkpoint
        )
        trainer.evaluate(
            eval_dataset=validation_set,
            metric_key_prefix="validation"
        )

    if step == TrainingStep.PRETRAINING:
        torch.save(
            model.linear_projector.state_dict(),
            os.path.join(cfg.pretraining.training_args.output_dir, "linear_projector.pth"),
        )


@hydra.main(version_base=None, config_path="../conf", config_name="default")
def main(cfg : DictConfig):

    path_to_projector = None

    if cfg.get(TrainingStep.PRETRAINING) is not None:
        train_model(
            step=TrainingStep.PRETRAINING,
            cfg=cfg,
        )
        path_to_projector = os.path.join(cfg.pretraining.training_args.output_dir, "linear_projector.pth")

    train_model(
        step="training",
        cfg=cfg,
        path_to_projector=path_to_projector,
    )


if __name__ == "__main__":
    main()
