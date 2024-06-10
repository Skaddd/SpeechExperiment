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


To run the Command Line Interface (CLI) for whisper real-time processing use the following commands :
```bash
$ poetry install
$ streaming --audio-folder ./data/EXAMPLES
```
For further information of specific commands see :

```bash
$ streaming -h

usage: streaming [-h] [--audio_file AUDIO_FILE] [--audio-folder AUDIO_FOLDER] [--model_size {tiny,base,small,medium,large-v2,large-v3}] [--min-chunk MIN_CHUNK] [--language LANGUAGE]
                 [--device {cpu,gpu}] [--compute-type COMPUTE_TYPE]

Description of parameters-- Multiprocessing is disabled by default

options:
  -h, --help            show this help message and exit
  --model_size {tiny,base,small,medium,large-v2,large-v3}, --model {tiny,base,small,medium,large-v2,large-v3}
                        Whisper model selector
  --min-chunk MIN_CHUNK, -chunk MIN_CHUNK
                        Min audio chunk size in seconds.
  --language LANGUAGE, -l LANGUAGE
                        Language spoken in audios
  --device {cpu,gpu}, -dev {cpu,gpu}
                        Define on which resources the transcription should be launched
  --compute-type COMPUTE_TYPE, -compute COMPUTE_TYPE
                        selecting compute precision during inference

Required parameters:
  --audio_file AUDIO_FILE, -f AUDIO_FILE
                        Audio file to transcribe in streaming way
  --audio-folder AUDIO_FOLDER, -folder AUDIO_FOLDER
                        Folder containing audio files in mp3/wav format
```
## Repositroy architecture: 

For a better understanding of the repository architecture, the repo tree below.
```
.
├── config
│   ├── finetune_whisper_parameters.yml
│   └── logging.yml
├── data
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
│       ├── run_fine-tuning.py
│       ├── train_whisper.py
│       └── utils
│           ├── helpers.py
│           ├── metric_utils.py
│           └── prep_utils.py
└── whisper-streaming
    ├── README.md
    └── src
        ├── transcript_stream.py
        └── utils
            ├── helper.py
            └── model.py

```