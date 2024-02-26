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
    <img src="https://i.ibb.co/RpcXwC5/SLAM-ASR-logo.png" alt="Logo" width="200" height="200">
  </a>

  <h3 align="center">SLAM-ASR</h3>

  <p align="center">
    Bringing audio to LLM: A fine-tuned LLM for ASR
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
- [ ] `SLAM` must accepts a batched tensor as input (currently expects a list representing a single audio sample)
- [ ] `SLAM.generate_transcript` method must be autoregressive and fully transcribe the input audio

### Training
- [ ] Write the data-preprocessing functions
- [ ] Write the training functions
- [ ] Overfeat the model on a subset of librispeech

### Distribution
- [ ] Write a proper README
- [ ] Upload model to Hugging-Face
- [ ] Create a Hugging-Face Space for the model
- [ ] Write a blog post presenting the projects and its inner working
- [ ] Record a video reproducing the projects
- [ ] Present the project over twitter
