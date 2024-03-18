import torch

from SLAM import SLAM
from evaluate import load
from safetensors import safe_open
from datasets import load_dataset


def map_to_pred(batch, model):
    audio = [sample["array"] for sample in batch["audio"]]
    audio = [torch.tensor(_audio, dtype=torch.float32, device=model.decoder.model.device) for _audio in audio]

    with torch.no_grad():
        predicted_ids = model.generate_transcript(audio)

    transcript = model.decoder.tokenizer.batch_decode(predicted_ids[:, :-1])

    batch["prediction"] = transcript

    return batch


def main(path_to_model):

    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"

    librispeech_test_clean = load_dataset("librispeech_asr", "all", split="test.other")

    model = SLAM(decode_name="abacaj/phi-2-super").to(device_to_use)

    tensors = {}
    with safe_open(path_to_model, framework="pt", device=device_to_use) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    model.load_state_dict(tensors)
    model.eval()

    # TODO: use processor
    # TODO: evaluate not inly on librispeech for ASR

    result = librispeech_test_clean.map(
        map_to_pred,
        fn_kwargs={"model": model},
        batched=True,
        batch_size=1,
    )
    labels = [text.capitalize() + "." for text in result["text"]]

    wer = load("wer")
    # ~1h
    print(100 * wer.compute(references=labels, predictions=result["prediction"]))
    # 6.673289647135243


if __name__ == "__main__":
    main(
        path_to_model="/scratch/SLAM-ASR-outputs/model/checkpoint-100000/model.safetensors",
    )

