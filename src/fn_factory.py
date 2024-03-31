import os
import functools

from typing import List


def _access_value(dict_to_access, key_as_str):
    if "." in key_as_str:
        return functools.reduce(lambda d, k: d[k], key_as_str.split('.'), dict_to_access)

    return dict_to_access


def is_shorter_than_model_max_length(columns: List[str], dataset, tokenizer, safe_padding: int = 0, **kwargs):
    model_max_length = tokenizer.model_max_length

    for column in columns:
        dataset = dataset.filter(
            lambda sample: len(_access_value(sample, column)) < model_max_length - safe_padding,
            num_proc=max(1, os.cpu_count() - 1),
        )

    return dataset


def is_longer_than(columns: List[str], value, dataset, **kwargs):

    for column in columns:
        dataset = dataset.filter(
            lambda sample: len(_access_value(sample, column)) > value,
            num_proc=max(1, os.cpu_count() - 1),
        )

    return dataset
