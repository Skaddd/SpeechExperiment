from datasets import load_dataset, concatenate_datasets, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
)
import logging

logger = logging.getLogger(__name__)


def load_whisper_utils(whisper_model_id: str):

    whisper_tokenizer = WhisperTokenizerFast.from_pretrained(
        whisper_model_id, language="English", task="transcribe"
    )

    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
        whisper_model_id
    )

    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_id)

    return whisper_tokenizer, whisper_feature_extractor, whisper_processor


def prepare_hf_dataset_whisper(
    raw_hf_dataset,
    whisper_tokenizer,
    whisper_feature_extractor,
    merged_train_val_bool: bool = True,
    useless_hf_columns: list[str] = [
        "age_group",
        "gender",
        "accent",
        "domain",
        "country",
        "duration",
        "speaker_id",
        "path",
        "audio_id",
    ],
    transcription_column: str = "transcript",
    audio_column_name: str = "audio",
    sampling_rate_whsiper: int = 16000,
):

    processed_hf_dataset = raw_hf_dataset.remove_columns(useless_hf_columns)

    logger.debug(
        "--- Ensuring the sampling rate corresponds "
        + f"to the one used by Whisper : {sampling_rate_whsiper} ---"
    )
    processed_hf_dataset = processed_hf_dataset.cast_column(
        audio_column_name, Audio(sampling_rate=sampling_rate_whsiper)
    )
    if merged_train_val_bool:
        processed_hf_dataset["train"] = concatenate_datasets(
            [
                processed_hf_dataset["train"],
                processed_hf_dataset["validation"],
            ]
        )

    def prepare_training_audios(
        data_batch,
        whisper_feature_extractor,
        whisper_tokenizer,
        transcription_column: str,
    ):

        audio_info = data_batch["audio"]

        data_batch["inputs_features"] = whisper_feature_extractor(
            audio_info["array"], sampling_rate=audio_info["sampling_rate"]
        )["input_features"]

        data_batch["labels"] = whisper_tokenizer(
            data_batch[transcription_column]
        )["input_ids"]

        return data_batch

    processed_hf_dataset = processed_hf_dataset.map(
        prepare_training_audios,
        batched=True,
        batch_size=50,
        fn_kwargs={
            "whisper_feature_extractor": whisper_feature_extractor,
            "whisper_tokenizer": whisper_tokenizer,
            "transcription_column": transcription_column,
        },
    )

    return processed_hf_dataset


def load_and_prepare_afrispeech_hf_dataset(
    whisper_feature_extractor,
    whisper_tokenizer,
    hf_dataset_args: dict,
    huggingface_dataset_name: str = "tobiolatunji/afrispeech-200",
    subsample_selector: str = "isizulu",
):
    logger.info(
        f"--- Loading HugginFace Dataset  : {huggingface_dataset_name} ---"
    )
    afrispeech_dataset = load_dataset(
        huggingface_dataset_name, subsample_selector, **hf_dataset_args
    )

    logger.info("-- Processing the dataset for Whisper fine-tuning ---")
    afrispeech_dataset_processed = prepare_hf_dataset_whisper(
        raw_hf_dataset=afrispeech_dataset,
        whisper_feature_extractor=whisper_feature_extractor,
        whisper_tokenizer=whisper_tokenizer,
    )
    return afrispeech_dataset_processed


if __name__ == "__main__":

    tokenizer, feature_extractor, processsor = load_whisper_utils(
        "openai/whisper-tiny"
    )
    processed_afrispeech = load_and_prepare_afrispeech_hf_dataset(
        whisper_feature_extractor=feature_extractor,
        whisper_tokenizer=tokenizer,
        hf_dataset_args={"trust_remote_code": True},
        huggingface_dataset_name="tobiolatunji/afrispeech-200",
        subsample_selector="isizulu",
    )
