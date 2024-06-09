import logging
import os

import yaml

from data_preparation import (load_and_prepare_afrispeech_hf_dataset,
                              load_whisper_utils)
from eval_whisper import eval_checkpoint_model
from train_whisper import load_whisper_checkpoint, train_model
from utils.helpers import setup_logging
from utils.prep_utils import DataCollatorSpeechSeq2SeqWithPadding

logger = logging.getLogger(__name__)
setup_logging()


def load_train_config():

    with open(
        os.path.join("config", "finetune_whisper_parameters.yml"), "r"
    ) as conffile:
        config_params = yaml.load(conffile, Loader=yaml.SafeLoader)

    return config_params


if __name__ == "__main__":

    config_params = load_train_config()
    tokenizer, feature_extractor, processor = load_whisper_utils(
        config_params["model_id"]
    )
    # Loading and processing dataset
    processed_afrispeech = load_and_prepare_afrispeech_hf_dataset(
        whisper_feature_extractor=feature_extractor,
        whisper_tokenizer=tokenizer,
        hf_dataset_args={"trust_remote_code": True},
        huggingface_dataset_name="tobiolatunji/afrispeech-200",
        subsample_selector="isizulu",
    )
    # Post processing - Padding inputs for standardize length.
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_params = config_params["whisper_training"]

    # Loading Whisper checkpoint.
    whisper_checkpoint = load_whisper_checkpoint(
        whisper_model_id=config_params["model_id"],
        lora_config_parameters=training_params["lora_config"],
    )
    # Using Trainer to Fine-tune Whisper
    train_model(
        whisper_checkpoint=whisper_checkpoint,
        seqtoseq_training_args=training_params["training_args"],
        data_collator=data_collator,
        training_dataset=processed_afrispeech["train"],
        val_dataset=processed_afrispeech["val"],
    )
    eval_checkpoint_model(
        checkpoint_model=whisper_checkpoint,
        validation_dataset=processed_afrispeech["test"],
        data_collator=data_collator,
        whisper_processor=processor,
    )
