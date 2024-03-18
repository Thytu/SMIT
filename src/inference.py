import torch
import torchaudio

from SLAM import SLAM
from safetensors import safe_open
from datasets import load_from_disk


def __load_weights(model, path_to_weights):
    tensors = {}
    with safe_open(path_to_weights, framework="pt", device=model.decoder.model.device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    model.load_state_dict(tensors)

    return model


def infer_over_audio(
    path_to_model: str,
    save_audio_sample: bool = False,
):

    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = __load_weights(SLAM(decode_name="abacaj/phi-2-super"), path_to_model)
    model.eval().to(device=device_to_use)

    validation_set = load_from_disk("outputs/dataset/")['validation']

    for i, sample in enumerate(iter(validation_set)):
        if i >= 10 and sample["inputs"].get("raw_audio") is not None:
            break

    raw_speech = sample["inputs"]["raw_audio"]
    labels = model.decoder.tokenizer.decode(sample["labels"])

    if save_audio_sample:
        torchaudio.save("sample.wav", torch.tensor(raw_speech).unsqueeze(0), format="wav", sample_rate=16_000)

    raw_speech = torch.tensor(raw_speech, device=device_to_use)

    input_ids = model.generate_transcript(raw_speech)

    transcript = model.decoder.tokenizer.batch_decode(input_ids)

    print(f"{transcript=}")
    print(f"{labels=}")


def infer_over_instruction(
    path_to_model: str,
):

    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = __load_weights(SLAM(decode_name="abacaj/phi-2-super"), path_to_model)
    model.eval().to(device=device_to_use)

    tokenizer = model.decoder.tokenizer


    messages = [
        {"role": "user", "content": "Hello, what's the capital for France? And what's french people favorite meal?"}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.decoder.model.device)
    input_ids_cutoff = inputs.size(dim=1)

    with torch.no_grad():
        generated_ids = model.decoder.model.generate(
            input_ids=inputs,
            use_cache=True,
            max_new_tokens=512,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    completion = tokenizer.decode(
        generated_ids[0][input_ids_cutoff:],
        skip_special_tokens=True,
    )

    print(completion)


if __name__ == "__main__":
    infer_over_audio(
        path_to_model="/scratch/SLAM-ASR-outputs/model/checkpoint-51000/model.safetensors",
        save_audio_sample=True,
    )

    infer_over_instruction(
        path_to_model="/scratch/SLAM-ASR-outputs/model/checkpoint-51000/model.safetensors",
    )
