# Whisper-finetuning

For this assignement the idea was to fine-tune Whisper on a specific language or accent. First the dataset selected was [AfriSpeech](https://huggingface.co/datasets/tobiolatunji/afrispeech-200). A dataset containing more than 200hours of speech corpus of general domain English **accented ASR**. Even though the *Whisper* model show nearly flawless performances on English language it can lack on specific accents that are not massively represented in its training dataset.


For simplicity and being able to actually run the code, i focused on `small`and `tiny`versions of `whisper` for results.

This time the HuggingFace implmentation was preferred as it gives more liberty for fine-tuning and optimization.


## Fine-Tuning and Optimizations 

Several optimizations were performed :
- First i used `bitsandbytes` library to use quantization and efficient reduce computation and almost keeping same performances.


Fr the fine-tuning part i mostly relied on [PeFT library](https://huggingface.co/docs/peft/index) (Parameter-Efficient Fine-Tuning) that contains state-of-the-art techniques to reduce trainable parameters during fine-tuning and effectively reduce the fine-tuning process.

The technique chosen was [LoRA](https://huggingface.co/docs/peft/package_reference/lora), even though it's a simple technique it has proven great results. The idea behind is simply to decompose attention layers in Transformer Models and greatly reduce the number of parameters that need to be fine-tuned. 

For our usecase :

```bash
trainable params: 589,824 || all params: 38,350,464 || trainable%: 1.5380
```

Parameters inside LoRa could have been extensively fine-tuned, however great results  were obtained with a simple `LoraConfig`

FOr a better reability a `conf`file is accessible, containing most of the hyperparameters used
[config.yml](/config/finetune_whisper_parameters.yml)


## Final Results

Finally, when working on a subsample of the global dataset (`isizulu language`), for the small model we managed to obtain : 

```bash
# after fine-tuning :
{'raw_wer_score': 33.35369578497251, 'normalised_wer_score': 29.703569267997583}

#before fine-tuning
{'raw_wer_score': 40.45610567911934, 'normalised_wer_score': 37.149874217298127}
``̀

