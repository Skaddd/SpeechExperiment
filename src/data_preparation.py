from datasets import load_dataset, concatenate_datasets, Audio, Dataset
from typing import Any
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
    PreTrainedTokenizerBase,
    FeatureExtractionMixin,
)
import logging


logger = logging.getLogger(__name__)


def load_whisper_utils(whisper_model_id: str) -> tuple[Any]:
    """Load Whisper Key components.

    Args:
        whisper_model_id (str): HuggingFace Whisper
        string identifier.

        example : ["openai/whisper-tiny", "openai/whisper-large-v3]

    Returns:
        tuple[Any]: whisper processor, feature extractor and tokenizer.
    """

    whisper_tokenizer = WhisperTokenizerFast.from_pretrained(
        whisper_model_id, language="English", task="transcribe"
    )

    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
        whisper_model_id
    )

    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_id)

    return whisper_tokenizer, whisper_feature_extractor, whisper_processor


def prepare_hf_dataset_whisper(
    raw_hf_dataset: Dataset,
    whisper_tokenizer: PreTrainedTokenizerBase,
    whisper_feature_extractor: FeatureExtractionMixin,
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
) -> Dataset:
    """Preprocess HF dataset to follow whisper requirements.

    This function aims at preprocessing any huggingface Dataset in order
    to make it prepared for any work on Whisper.
    - Useless columns are removed
    - Sampling Rate downsampled to 16kHz.
    - Raw Audio inputs transformed to Mel-Spectrogram (feature extractor).
    - Transcriptions are encoded (tokenizer).
    Args:
        raw_hf_dataset (Dataset): Raw HuggingFace dataset selected.
        whisper_tokenizer (PreTrainedTokenizerBase): Whisper Fast Tokenizer..
        whisper_feature_extractor (FeatureExtractionMixin): Whisper Feature *
            Extractor pretrained.
        merged_train_val_bool (bool, optional): Whether training data should
          be augmented with validation data. Defaults to True.
        useless_hf_columns (list[str], optional): Metdata columns
            in the HF Dataset that are irrelevant.
            Defaults to [
            "age_group", "gender",
            "accent", "domain",
            "country", "duration",
            "speaker_id", "path", "audio_id", ].
        transcription_column (str, optional): Dataset column with
            reference transcriptions. Defaults to "transcript".
        audio_column_name (str, optional): Dataset column with audios specs.
            Defaults to "audio".
        sampling_rate_whsiper (int, optional): Sampling rate for Whisper.
            Defaults to 16000.

    Returns:
        Dataset: prepared HF dataset for fine-tuning Whisper.
    """

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
        data_batch: Any,
        whisper_feature_extractor: FeatureExtractionMixin,
        whisper_tokenizer: PreTrainedTokenizerBase,
        transcription_column: str,
    ):
        """Batch processing of samples inside Dataset.

        Helper function used with Dataset.map() utils
        to rapidly process raw samples for a Dataset and create
        new columns useful.
        Args:
            data_batch (_type_): Raw samples.
            whisper_feature_extractor (FeatureExtractionMixin): Feature
                Extractor to get Mel-Spectrogram.
            whisper_tokenizer (PreTrainedTokenizerBase): Tokenizer
                to encode transcriptions.
            transcription_column (str): transcriptions column.
        Returns:
            Any: processed samples
        """
        audio_info = data_batch["audio"]

        data_batch["inputs_features"] = whisper_feature_extractor(
            audio_info["array"], sampling_rate=audio_info["sampling_rate"]
        )["input_features"][0]

        data_batch["labels"] = whisper_tokenizer(
            data_batch[transcription_column]
        )["input_ids"]

        return data_batch

    processed_hf_dataset = processed_hf_dataset.map(
        prepare_training_audios,
        fn_kwargs={
            "whisper_feature_extractor": whisper_feature_extractor,
            "whisper_tokenizer": whisper_tokenizer,
            "transcription_column": transcription_column,
        },
    )

    return processed_hf_dataset


def load_and_prepare_afrispeech_hf_dataset(
    whisper_feature_extractor: FeatureExtractionMixin,
    whisper_tokenizer: PreTrainedTokenizerBase,
    hf_dataset_args: dict,
    huggingface_dataset_name: str = "tobiolatunji/afrispeech-200",
    subsample_selector: str = "isizulu",
) -> Dataset:
    """Process AfriSpeech HuggingFace Dataset.

    Args:
        whisper_feature_extractor (FeatureExtractionMixin): Whisper
          Feature Extractor.
        whisper_tokenizer (PreTrainedTokenizerBase): Whisper Tokenizer.
        hf_dataset_args (dict): Addtional HF dataset args.
        huggingface_dataset_name (str, optional): Name for the Dataset.
            Defaults to "tobiolatunji/afrispeech-200".
        subsample_selector (str, optional): subsample selector.
            Defaults to "isizulu".

    Returns:
        Dataset: Processing HF dataset
    """
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
