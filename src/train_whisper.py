import logging
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

from utils.helpers import display_linear_modules
from utils.prep_utils import SavePeftModelCallback

logger = logging.getLogger(__name__)


def load_whisper_checkpoint(
    whisper_model_id: str, lora_config_parameters: dict
):
    """Prepare Whisper model for Fine-Tuning with PEFT and LoRa.

    This function aims at loading a whisper checkpoint for
    fine-tuning purposes. Then, to optimizer the fine-tuning process
    Quantization and PeFT techniques are used to make it faster.
    LoRa technique is used as it has shown great results.
    Args:
        whisper_model_id (str): whisper model to used.
        lora_config_parameters (dict): LoRa parameters that
        should be used.

    Returns:
        _type_: Prepared model
    """

    logger.info("--- Quantization the model parameters to INT8 ---")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_model_id, quantization_config=quantization_config
    )
    # Swap back layers that are going to be trained ?
    whisper_model = prepare_model_for_kbit_training(whisper_model)

    logger.debug("--- PEFT : Forcing embedding inputs to require grads ---")

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    whisper_model.get_input_embeddings().register_forward_hook(
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
    whisper_checkpoint,
    seqtoseq_training_args: dict[str, Any],
    data_collator: Any,
    training_dataset: Dataset,
    val_dataset: Dataset,
) -> None:
    """Launch HuggingFace Trainer for Fine-tuning Whisper.

    This function launch a huggingFace Trainer and
    use a PeFT callback to follow performances during training.
    Args:
        whisper_checkpoint (_type_): whisper checkpoint.
        seqtoseq_training_args (dict[str, Any]): Training args.
        training_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
    """

    logger.info("--- Launching Whisper Fine-Tuning ---")

    logger.info(
        "--- Results will be saved in "
        + f"{seqtoseq_training_args['output_dir']} ---"
    )
    seqtoseq_training_args.update(
        {
            "remove_unused_columns": False,
            "label_names": ["labels"],
        }
    )
    training_args = Seq2SeqTrainingArguments(**seqtoseq_training_args)

    peft_callback = SavePeftModelCallback()

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=whisper_checkpoint,
        train_dataset=training_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[peft_callback],
    )
    whisper_checkpoint.config.use_cache = False
    trainer.train()
