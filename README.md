<div id="top"></div>

<br />
<div align="center">

  
  <a href="https://github.com/Thytu/SMIT/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/Thytu/SMIT.svg?style=for-the-badge&color=blue" alt="Logo">
  </a>
  <a href="https://github.com/Thytu/SMIT/network/members">
    <img src="https://img.shields.io/github/forks/Thytu/SMIT.svg?style=for-the-badge&color=blue" alt="Logo">
  </a>
  <a href="https://github.com/Thytu/SMIT/stargazers">
    <img src="https://img.shields.io/github/stars/Thytu/SMIT.svg?style=for-the-badge&color=yellow" alt="Logo">
  </a>
  <a href="https://github.com/Thytu/SMIT/issues">
    <img src="https://img.shields.io/github/issues/Thytu/SMIT.svg?style=for-the-badge&" alt="Logo">
  </a>
  <a href="https://github.com/Thytu/SMIT/pulls">
    <img src="https://img.shields.io/github/issues-pr/Thytu/SMIT.svg?style=for-the-badge" alt="Logo">
  </a>

  <br/>
  <br/>
  
  <a href="https://github.com/Thytu/SMIT/network/members">
    <img src="https://repository-images.githubusercontent.com/763042457/1d2a98b1-5e63-4416-9da2-6b432fcb0726" alt="Logo">
  </a>

  <h3 align="center" style="font-size: 200%">SMIT</h3>

  <p align="center">
    <b> A Simple Modality Integration Tool </b>
    <br />
    <br />
    <a href="#getting-started"><strong>Explore the docs</strong></a>
    <br />
    <br />
    <a href="#about-the-project">More about SMIT</a>
    · <a href="https://github.com/Thytu/SMIT/issues">Report Bug or Request Feature</a>
  </p>
</div>

<br/>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#how-it-works">How it works</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<br/>


## About The Project

SMIT is a versatile tool designed to streamline the integration of audio modality into your LLMs. Currently, SMIT exclusively supports audio as a new modality. However, our goal is to expand its capabilities to accommodate any new modality seamlessly. We welcome contributions from the open-source community to help us achieve this aim.


<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

Welcome to SMIT! Follow these simple steps to get started:

Begin by cloning the SMIT repository to your local machine using Git:
```sh
git clone https://github.com/Thytu/SMIT/
cd SMIT
```

We highly recommend using a virtual environment to manage dependencies and prevent conflicts. Create and activate a virtual environment using your preferred tool (e.g., virtualenv, conda):

```sh
# Example using virtualenv
virtualenv venv
source venv/bin/activate
```

Once inside the project directory and your virtual environment is activated, install the required dependencies listed in requirements.txt using pip:

```sh
pip install -r requirements.txt
```

### Run the Example

You can quickly run the default example provided in SMI by executing the following command:

```sh
python src/main.py
```

This will train the amazing [abacaj/phi-2-super](https://huggingface.co/abacaj/phi-2-super/tree/main) model to do ASR using the `librispeech_asr` dataset and [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft) as speech encoder, reproducing the [Thytu/phi-2-audio-super](https://huggingface.co/Thytu/phi-2-audio-super/tree/main) model.

> [!IMPORTANT]
> It's essential to ensure a minimum of 30GB of available VRAM to execute this command successfully.
>  For users with >=80GB of VRAM, it's recommended to deactivate quantization while decreasing the batch size to expedite the training process. You can achieve this by running:
> ```
> python src/main.py ~model.decoder.quantization_config ++training.training_args.per_device_train_batch_size=1
> ```

### Customize Your Model

To customize your own Language Model (LLM), create a [configuration file](docs/config-file.md). You can use the provided [config file template](config/default.yaml) as a starting point. Then, use [Hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) to provide your configuration file:

```sh
python src/main.py model=my_config
```

Hydra offers extensive options for parameter overriding, allowing you to tailor the model according to your specific requirements. Refer to [Hydra documentation](https://hydra.cc/docs/intro/) for more details on customization options.

### Inference

Once your model is trained, you can effortlessly load it for inference:
```py
model = SMIT.from_pretrained("path_to_your_safetensor")
```

For inference tasks, you can utilize the `generate` method:
```py
model.generate("Tell me how to add a modality to my model")
```

To employ the `generate` method with multiple modalities, follow this approach:
```py
model.generate(
    prompt=[
        "Tell me how to add a modality to my model",
        "Transcribe this audio from speech to text {audio}",
    ],
    raw_speech=[None, you_audio],
)
```

> [!NOTE]
> When providing multiple prompts, ensure that the length of `raw_speech` matches the length of `prompt`.



<p align="right">(<a href="#top">back to top</a>)</p>


## How it works

SMIT simplifies the process of enhancing your LLM with audio capabilities, following the principles outlined in the [this paper](https://arxiv.org/abs/2402.08846). By linking a speech encoder to an decoder using a trainable linear projector adding to your LLM the audio modality. SLMA automates the integration process by making it as easy as configuring a single file.

To use SMIT, simply define your desired configurations in the provided config file, it will then handle the rest, seamlessly incorporating the audio modality into your models.

![Untitled-2022-08-10-1416](https://github.com/Thytu/SMIT/assets/43698357/7a4843d8-d283-4d3b-ab7f-1f4ba0199e4b)

<p align="right">(<a href="#top">back to top</a>)</p>


## Contributing

There are mutliple ways to contribute to that projects, either regarding the UX (i.e doc / even making the example faster) or regarding the core product itself (i.e handling Vision modality).
Any contributions you make are **greatly appreciated**, if you have a suggestion that would make this better feel free to tell me :D You can also check the [open issues](https://github.com/Thytu/SMIT/issues) for more things to improve.

Don't forget to give the project a star! 🌟 Thanks again!

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments

This project draws significant inspiration from the [An Embarrassingly Simple Approach for LLM with Strong ASR Capacity](https://arxiv.org/pdf/2402.08846.pdf) paper. I thank the authors for sharing their expertise. Huge thanks to the CoolKids for their  help in debugging some pesky issues I ran into. And last but definitely not the least, a massive thank you to Oursin – this project simply wouldn't exist without you!

<p align="right">(<a href="#top">back to top</a>)</p>


## Contact

Hey, I'm Valentin De Matos, passionate about AI and always working on some new side project.

You can reach me out at vltn.dematos@gmail.com and if you want more information you can always
- Check my website [thytu.com](https://thytu.com/)
- Follow me on twitter [@ThytuVDM](https://twitter.com/ThytuVDM)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[issues]: https://img.shields.io/github/issues/Thytu/SMIT
