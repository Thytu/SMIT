<div id="top"></div>


[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![PUll Request][pr-shield]][pr-url]
[![MIT License][license-shield]][license-url]

<br />
<div align="center">
  <a href="https://github.com/Thytu/SMIT">
    <img src="https://i.ibb.co/cY65gkZ/SMIT-logo-v1.webp" alt="Logo" width="80%" height="80%">
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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#todo-list">Roadmap</a></li>
  </ol>
</details>

<br/>


## About The Project

SMIT is a versatile tool designed to streamline the integration of audio modality into your LLMs. Currently, SMIT exclusively supports audio as a new modality. However, our goal is to expand its capabilities to accommodate any new modality seamlessly. We welcome contributions from the open-source community to help us achieve this aim.


<p align="right">(<a href="#top">back to top</a>)</p>


## Usage

SMIT simplifies the process of enhancing your LLM with audio capabilities, following the principles outlined in the [this paper](https://arxiv.org/abs/2402.08846). By linking a speech encoder to an decoder using a trainable linear projector adding to your LLM the audio modality. SLMA automates the integration process by making it as easy as configuring a single file.

To use SMIT, simply define your desired configurations in the provided config file, it will then handle the rest, seamlessly incorporating the audio modality into your models.

![Untitled-2022-08-10-1416](https://github.com/Thytu/SMIT/assets/43698357/7a4843d8-d283-4d3b-ab7f-1f4ba0199e4b)

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

Welcome to SMIT! Follow these simple steps to get started:

**Clone the Repository**: Begin by cloning the SMIT repository to your local machine using Git:
```sh
git clone https://github.com/Thytu/SMIT/
cd SMIT
```

**Set Up Virtual Environment**: We highly recommend using a virtual environment to manage dependencies and prevent conflicts. Create and activate a virtual environment using your preferred tool (e.g., virtualenv, conda):
```sh
# Example using virtualenv
virtualenv venv
source venv/bin/activate
```

**Install Dependencies**: Once inside the project directory and your virtual environment is activated, install the required dependencies listed in requirements.txt using pip:

```sh
pip install -r requirements.txt
```

**Run the Example**: You can quickly run the default example provided in SMI by executing the following command:

```sh
python src/main.py
```

This will train the amazing [abacaj/phi-2-super](https://huggingface.co/abacaj/phi-2-super/tree/main) model to do ASR using the `librispeech_asr` dataset and [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft) as speech encoder, reproducing the [Thytu/phi-2-audio-super](https://huggingface.co/Thytu/phi-2-audio-super/tree/main) model. (Requires at least 40Go of VRAM)

**Customize Your Model**: To customize your own Language Model (LLM), create a [configuration file](docs/config-file.md). You can use the provided [config file template](config/default.yaml) as a starting point. Then, use [Hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) to provide your configuration file:

```sh
python src/main.py model=my_config
```

**Advanced Configuration**: Hydra offers extensive options for parameter overriding, allowing you to tailor the model according to your specific requirements. Refer to [Hydra documentation](https://hydra.cc/docs/intro/) for more details on customization options.


<p align="right">(<a href="#top">back to top</a>)</p>

## Contributing

There are mutliple ways to contribute to that projects, either regarding the UX (i.e doc / even making the example faster) or regarding the core product itself (i.e handling Vision modality).
Any contributions you make are **greatly appreciated**, if you have a suggestion that would make this better feel free to tell me :D You can also check the <a href="#todo-list">TODO list</a> for more things to improve.

Don't forget to give the project a star! 🌟 Thanks again!

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments

This project draws significant inspiration from the [An Embarrassingly Simple Approach for LLM with Strong ASR Capacity](https://arxiv.org/pdf/2402.08846.pdf) paper. I thank the authors for sharing their expertise. Huge thanks to the CoolKids for their  help in debugging some pesky issues I ran into. And last but definitely not the least, a massive thank you to Oursin – this project simply wouldn't exist without you!

<p align="right">(<a href="#top">back to top</a>)</p>


## Contact

Hey, I'm Valentin De Matos, passionate about AI and always working on some new side project.

You can reach me out at vltn.dematos@gmail.com and if you want more information you can always
- Check my website [@hytu.com](https://thytu.com/)
- Follow me on twitter [@ThytuVDM](https://twitter.com/ThytuVDM)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/Thytu/SMIT.svg?style=for-the-badge&color=blue
[contributors-url]: https://github.com/Thytu/SMIT/graphs/contributors
[pr-shield]: https://img.shields.io/github/issues-pr/Thytu/SMIT.svg?style=for-the-badge
[pr-url]: https://github.com/Thytu/SMIT/pulls
[issues]: https://img.shields.io/github/issues/Thytu/SMIT
[forks-shield]: https://img.shields.io/github/forks/Thytu/SMIT.svg?style=for-the-badge&color=blue
[forks-url]: https://github.com/Thytu/SMIT/network/members
[stars-shield]: https://img.shields.io/github/stars/Thytu/SMIT.svg?style=for-the-badge&color=yellow
[stars-url]: https://github.com/Thytu/SMIT/stargazers
[issues-shield]: https://img.shields.io/github/issues/Thytu/SMIT.svg?style=for-the-badge&
[issues-url]: https://github.com/Thytu/SMIT/issues
[license-shield]: https://img.shields.io/github/license/Thytu/SMIT.svg?style=for-the-badge&color=indigo
[license-url]: https://github.com/Thytu/SMIT/blob/master/LICENSE

## TODO List

### Model
- [X] `SMIT` must accepts a batched tensor as input (currently expects a list representing a single audio sample)
- [ ] `SMIT.encoder` should not have to use the fined-tunable version of `hubert-large-ls960`
- [X] `SMIT.generate_transcript` method must be autoregressive and fully transcribe the input audio

### Data
- [X] Write the data-preprocessing functions
- [X] data-preprocessing must capitalize the text properly (not all cap)
- [X] Export processed dataset locally to be loaded at training time
- [X] Reduce number of samples for continous training
- [X] Link hydra to data_handler
- [X] Support other datasets
- [ ] Create audio instruct dataset

### Training
- [X] Write the training functions
- [X] Overfeat the model on a subset of librispeech
- [X] Train the model on the full set of librispeech
- [X] Fix why the model doesn't procudes EOS (or issue on inference fn?)
- [X] Padding should be set to max(len_of_inputs) instead of always 2048
- [X] Pre-training on projector
- [X] Support other LLM
- [X] Support other speech encoder

### Evaluation
- [X] Evaluate on librispeech
- [ ] Check if it impacts phi2 results on OpenLLM-Leaderboard

### Distribution
- [ ] Reduce RAM usage
- [X] Reduce VRAM usage
- [ ] Write a proper README
- [ ] Write a "How to use the model" doc with everything required for inference (i.e using feature_extractor)
- [ ] Wrape in docker
- [X] Upload model to Hugging-Face
- [ ] Create a Hugging-Face Space for the model
- [ ] Record a Video reproducing the projects
- [ ] Share over HF's discord in i-made-this channel
- [ ] Write a blog post presenting the projects and its inner working
- [ ] Present the project over twitter

<p align="right">(<a href="#top">back to top</a>)</p>
