from utils.prep_utils import (
    DataCollatorSpeechSeq2SeqWithPadding,
    SavePeftModelCallback,
)
from data_preparation import load_whisper_utils
from utils.helpers import display_linear_modules

from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


import torch
import logging

logger = logging.getLogger(__name__)


def load_whisper_checkpoint(
    whisper_model_id: str, lora_config_parameters: dict
):

    logger.info("--- Quantization the model parameters to INT8 ---")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # comprendre cette methode
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_model_id, quantization_config=quantization_config
    )
    # Swap back layers that are going to be trained ?
    whisper_model = prepare_model_for_kbit_training(whisper_model)

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    whisper_model.model.encoder.conv1.register_forward_hook(
        make_inputs_require_grad
    )

    display_linear_modules(whisper_model)
    config = LoraConfig(**lora_config_parameters)

    whisper_model = get_peft_model(whisper_model, config)
    logger.info(
        "--- Trainable parameters of Whisper after Freezing : "
        + f"{whisper_model.print_trainable_parameters()} ---"
    )

    return whisper_model


def train_model(
    whisper_loaded_model,
    seqtoseq_training_args,
    training_dataset,
    val_dataset,
) -> None:

    seqtoseq_training_args.update(
        {
            "output_dir": "mateo/results",
            "remove_unused_columns": False,
            "label_names": ["labels"],
        }
    )
    training_args = Seq2SeqTrainingArguments(**seqtoseq_training_args)

    peft_callback = SavePeftModelCallback()

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=whisper_loaded_model,
        train_dataset=training_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        callbacks=[peft_callback],
    )
    whisper_loaded_model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":

    tokenizer, feature_extractor, processor = load_whisper_utils(
        "openai/whisper-tiny"
    )
    lora_config = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
    }

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    load_whisper_checkpoint("openai/whisper-tiny", lora_config)
