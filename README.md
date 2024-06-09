# SpeechExperiment

This repository explore 3 typical usescases that leverages on ASR technologies and techniques :


1. [**Build a VAD**](./whisper-finetuning/README.md): *Implement and evaluate/test a VAD system from scratch.  The system implemented is also evaluted by carefully selecting relevant metrics and datasets.*
2. [**Fine-tune a whisper model**](./vad-implementation/README.md): *Implement a training script with the goal to fine-tune a whisper based on a specific language of your choice. The script must be optimised for a fast inference time on GPU*

3. [**Whisper/Canary 1B streaming**](./whisper-streaming/README.md): *Implement a real-time transcription system with the least possible delay and the highest possible perfomance. Then evaluate the system on audio files and Then implement multi-processing/multi-threaded scripts*
 
Each pipeline contains it's own `README.md` file for deeper explanation.


## Basic usage

Once the repository is cloned and your env is created :

```bash
$ git clone https://github.com/Skaddd/SpeechExperiment.git
$ conda create -n pyspeech python=3.10.14
```

First install `poetry package` then install all dependencies :

```bash
$ pip install poetry
$ poetry install
```

> NOTE : GPU is highly advised for most of the pipelines and this package suppose that you have already correctly installed CUDA on you machine. For more information regarding CUDA installation refer to [Pytorch documentation](https://pytorch.org/get-started/locally/).



## Repositroy architecture: 



```
├── config
│   ├── finetune_whisper_parameters.yml
│   └── logging.yml
├── data
├── poetry.lock
├── pyproject.toml
├── README.md
├── results
├── vad-implementation
│   ├── README.md
│   └── src
├── weights
├── whisper-finetuning
│   ├── README.md
│   └── src
│       ├── data_preparation.py
│       ├── eval_whisper.py
│       ├── __pycache__
│       │   └── data_preparation.cpython-310.pyc
│       ├── run_fine-tuning.py
│       ├── train_whisper.py
│       └── utils
│           ├── helpers.py
│           ├── metric_utils.py
│           ├── prep_utils.py
│           └── __pycache__
│               ├── helpers.cpython-310.pyc
│               ├── metric_utils.cpython-310.pyc
│               └── prep_utils.cpython-310.pyc
└── whisper-streaming
    ├── README.md
    └── src
```