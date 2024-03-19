# How to use and write config file for SLIM

SLIM relies on config files to know which model to train and on what data.

## Model

Everything relative to the models to train.

### Encoder

Everything about the speech encoder to use (note that its weights are freezed and will not be trained).

| Param           | description                                         | Example                          |
|-----------------|-----------------------------------------------------|----------------------------------|
| `model_name`    | HF name of the speech encoder to use                | `facebook/hubert-large-ls960-ft` |
| `sampling_rate` | expected audio sampling rate for the speech encoder | `16000`                          |

### Decoder

Everything about the LLM to train.

| Param           | description                                         | Example                        |
|-----------------|-----------------------------------------------------|--------------------------------|
| `model_name`          | HF name of the LLM to use                                     | `abacaj/phi-2-super`                   |
| `audio_placeholder`   | How SLIM will know where to insert the audio into the prompt  | `{audio}` (default)                    |
| `prompt_template`     | LLM's expected instruct format                                | `"EOS_TOKEN[INST] {instruct} [/INST]"` |
| `quantization_config` | (optional) params for [BitsAndBytesConfig](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)  |                                |
| `peft` | (optional) params for [LoraConfig](https://huggingface.co/docs/peft/v0.9.0/en/package_reference/lora#peft.LoraConfig)  |                                |


## Data

List of dataset to train the model on, note that each dataset must be provided as follows:
```yaml
datasets:

  name_of_dataset:
    param: value
```

| Param             | description                                                                                                                                                                                                                                                                                                                                           | Example                             |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| `audio_column`    | (optional) Name of the column containing the audio within the dataset. None if not provided.                                                                                                                                                                                                                                                          | `audio`                             |
| `label_column`    | Name of the column containing the expected output.                                                                                                                                                                                                                                                                                                    | `output`                            |
| `instruct_column` | Name of the column containing the input instruct. If `audio_column` is not None, it must contains `audio_placeholder`.                                                                                                                                                                                                                                | `input`                             |
| `instruct`        | (optional) Overwrites the values in `instruct_column` or create one if not present.                                                                                                                                                                                                                                                                   | `Transcribe speech to text {audio}` |
| `subset`          | (optional) param for [load_dataset](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.load_dataset)                                                                                                                                                                                                          | `all`                               |
| `splits`          | List of the three sets (`train`, `test`, `validation`) used for training, passed as it to [load_dataset](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.load_dataset) as [split](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.load_dataset.split) argument. |                                     |
| `filters`         | (optional) Allows the filtering of the dataset, see the `More on filters` section.

### More on filters

Filters are a way to allow data filtering (i.e removing column longer that the model's context length), it relies on a suite of pre-written function that can be chained to formate a filter provided to [`Dataset.filter`](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/main_classes#datasets.Dataset.filter).

Available filters can be founded in [fn_factory.py](../src/fn_factory.py), feel free to add any filters that you found missing.


## Pretraining

Same as <a href="#training"><strong>Training</strong></a>, note that pretraining is fully optinal and can't be omitted.

## Training

Parameters used to determine training behavior such as batch size, training step and others.

| Param                     | description                                                                                                                             | Example       |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `wandb_project_name`      | (optional) how to name the wandb project                                                                                                | SLIM-training |
| `early_stopping_patience` | param for [EarlyStoppingCallback](https://huggingface.co/docs/transformers/main_classes/callback#transformers.EarlyStoppingCallback)    | `5`           |
| `resume_from_checkpoint`  | (optional) If provided, the training will resume to the provided checkpoint path                                                        | `some/path/`  |
| `training_args`           | Params for [TrainingArguments](https://huggingface.co/docs/transformers/v4.39.0/en/main_classes/trainer#transformers.TrainingArguments) |               |
