# Whisper Streaming
This assignement aimed at implementing a real-time transcription system with the best possible performances : 
- Least delay
- Highest WER
 
 Then developing an evaluation pipeline on any given dataset, to evaluate and compare the system produced. Finally to answer large volumes of calls the last objective was to enhance the current system to make it work on multiple audios at the same time using `multi-processing/multi-threading` routines.


First the [faster-whisper](https://github.com/SYSTRAN/faster-whisper/tree/master) implementation was rapidly selected for its performances and its possibility to run on CPU ! Indeed, having a working system was essential for testing purposes, thus i also focused on selecting an implementation would actually work on any given device.


> NOTE: **With this additional constraint, several real-time whisper implementation were not selected because of their GPU requirements.** [WhisperLive](https://github.com/collabora/WhisperLive/tree/main)


The implementation mostly relies on a specific parameter named `init_prompt` that allows one to provide addtional context during the inference time. Thus to rapidly describe the current implementation :

- The raw audio file is segmented into intermediate chunks of audio
- EAch chunk is appended incrementally to a buffer
- The model is then ran on the buffer, using initial_prompt by passing the previous result of this same step.
- Then a work is done to avoid overlapping in transcriptions.


## Runing The project

A rapid yet efficient CLI was developed leveraging on poetry simplicity. Thus anyone can try this implemetation of Real-time transcription :

```bash
# assuming you have an env with poetry installed
$ poetry install
$ streaming -h
$ streaming --audio-folder ./data/
```
Results will be displayed directly in the terminal and final transcriptions linked to every file transcripted will be saved  in a JSON format.


The actual project doesn't evaluate the performances of the implementation and doesn't used multi-processing. **However is set up to effortlessly implement an evaluation pipeline**.

For simplicity not all hyperparameters are accessible through the Command Line Interface.


## Improvements

Several improvements can be made rapidly :

1. First everything is inplace for metric evaluation, full trasncriptions are saved, we just have to download any dataset and launch a quick script on it to compute WER for performance and processing time for speed.
2. For the multi-processing part, it's actually the same, everything is inplace to run multiprocessing inference. Faster-Whisper works weirdly on CPU with the options `cpu_threads` and `num_works`. A simply yet efficient way to still be able to run multi instances is to use joblib delayed function :

```python
executor= Parallel(n_jobs=3)
do=delayed(multi_online_transcription)
tasks = (
    do(
        all_audios[i: i +batch_size],
        args,
        logfile
    )
    for i in range(0, len(all_audios), batch_size)
)
_=executor(tasks)
```

Few tweakings might be necessary, for example, the model should be instanciated inside the function called.

3. Performance enhancement and speed-up inference was not performed too, however faster-whisper is one of the fastest implementation for Whisper + it runs efficiently on CPUs. However tweaks were also possible working on other factors (such as `compute_type`).