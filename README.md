<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![PUll Request][pr-shield]][pr-url]
[![MIT License][license-shield]][license-url]


<br />
<div align="center">
  <a href="https://github.com/Thytu/SLAM-ASR">
    <img src="https://i.ibb.co/CvLbGX6/SLAM-ASR-logo-v2.png" alt="Logo" width="200" height="200">
  </a>

  <h3 align="center" style="font-size: 200%">SLAM-ASR</h3>

  <p align="center">
    <b> Bringing audio to LLM: A fine-tuned LLM for ASR</b>
    <br />
    <a href="#usage"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#about-the-project">View Demo</a>
    · <a href="https://github.com/Thytu/SLAM-ASR/issues">Report Bug</a>
    · <a href="https://github.com/Thytu/SLAM-ASR/issues">Request Feature</a>
  </p>
</div>

<br/>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<br/>


## About The Project

TODO: Presentation of this project

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

TODO: Guide through requirements

<p align="right">(<a href="#top">back to top</a>)</p>


## Usage

TODO: Guide on how to use the model
TODO: Guide on how to reproduce the model

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.\
If you have a suggestion that would make this better, please fork the repo and create a pull request.

Don't forget to give the project a star! 🌟 Thanks again!

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments

TODO: Cite Alibaba's paper

<p align="right">(<a href="#top">back to top</a>)</p>



## Contact

Valentin De Matos - [@ThytuVDM](https://twitter.com/ThytuVDM) - vltn.dematos@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/Thytu/SLAM-ASR.svg?style=for-the-badge
[contributors-url]: https://github.com/Thytu/SLAM-ASR/graphs/contributors
[pr-shield]: https://img.shields.io/github/issues-pr/Thytu/SLAM-ASR.svg?style=for-the-badge
[pr-url]: https://github.com/Thytu/SLAM-ASR/pulls
[issues]: https://img.shields.io/github/issues/Thytu/SLAM-ASR
[forks-shield]: https://img.shields.io/github/forks/Thytu/SLAM-ASR.svg?style=for-the-badge&
[forks-url]: https://github.com/Thytu/SLAM-ASR/network/members
[stars-shield]: https://img.shields.io/github/stars/Thytu/SLAM-ASR.svg?style=for-the-badge&
[stars-url]: https://github.com/Thytu/SLAM-ASR/stargazers
[issues-shield]: https://img.shields.io/github/issues/Thytu/SLAM-ASR.svg?style=for-the-badge&
[issues-url]: https://github.com/Thytu/SLAM-ASR/issues
[license-shield]: https://img.shields.io/github/license/Thytu/SLAM-ASR.svg?style=for-the-badge&
[license-url]: https://github.com/Thytu/SLAM-ASR/blob/master/LICENSE

## TODO List

### Model
- [X] `SLAM` must accepts a batched tensor as input (currently expects a list representing a single audio sample)
- [ ] `SLAM.encoder` should not have to use the fined-tunable version of `hubert-large-ls960`
- [X] `SLAM.generate_transcript` method must be autoregressive and fully transcribe the input audio

### Data
- [X] Write the data-preprocessing functions
- [X] data-preprocessing must capitalize the text properly (not all cap)
- [X] Export processed dataset locally to be loaded at training time
- [ ] Reduce number of samples for continous training
- [ ] Create audio instruct dataset

### Training
- [X] Write the training functions
- [X] Overfeat the model on a subset of librispeech
- [X] Train the model on the full set of librispeech
- [X] Fix why the model doesn't procudes EOS (or issue on inference fn?)
- [ ] Pre-training on projector
- [X] Padding should be set to max(len_of_inputs) instead of always 2048

### Evaluation
- [X] Evaluate on librispeech
- [ ] Check if it impacts phi2 results on OpenLLM-Leaderboard

### Distribution
- [ ] Use [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) to create a user CLI
- [ ] Write a proper README
- [ ] Write a "How to use the model" doc with everything required for inference (i.e using feature_extractor)
- [X] Upload model to Hugging-Face
- [ ] Create a Hugging-Face Space for the model
- [ ] Record a Video reproducing the projects
- [ ] Share over HF's discord in i-made-this channel
- [ ] Write a blog post presenting the projects and its inner working
- [ ] Present the project over twitter
