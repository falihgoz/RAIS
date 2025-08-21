# Rehearsal with Auxiliary-Informed Sampling (RAIS)

This repository contains the implementation of RAIS.

# Installation

Python==3.12.3

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```

# Dataset

Metadata for train, validation, and evaluation splits can be found in `dataset/`.

- **ASVspoof 2019 LA** ($\mathcal{E}_0$): Download [`LA.zip`](https://datashare.ed.ac.uk/handle/10283/3336).
- **VCC 2020** ($\mathcal{E}_1$): Combined datasets from
  - [Source 1](https://github.com/nii-yamagishilab/VCC2020-listeningtest)
  - [Source 2](https://github.com/nii-yamagishilab/VCC2020-database)
- **InTheWild** ($\mathcal{E}_2$): Download [`release_in_the_wild.zip`](https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa).
- **CFAD** ($\mathcal{E}_3$): Download all CFAD zip files from [Zenodo](https://zenodo.org/records/8122764).
- **OpenAI-LJSpeech** ($\mathcal{E}_4$):
  - Fake audios generated using the [OpenAI Text-to-Speech API](https://platform.openai.com/docs/guides/text-to-speech).
  - Bonafide samples from [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

# Usage

- Run command:

  ```bash
  python main.py --method rais
  ```

# Cite this work

TBD
