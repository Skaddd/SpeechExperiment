import logging

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from utils.metric_utils import compute_wer

logger = logging.getLogger(__name__)


def load_baseline_whisper_model(
    whisper_model_id: str, whisper_pipeline_args: dict = None
):
    """Load whisper baseline.

    Args:
        whisper_model_id (str): huggingface whisper model selected.
            example : ["openai/whisper-tiny]
        whisper_pipeline_args (dict, optional): additional args
         to the whisper pipeline.
        Defaults to None.

    Returns:
        _type_: Prepared whisper pipeline.
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(
        f"--- Model selected : {whisper_model_id} - with device : {device} ---"
    )

    try:

        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
    except OSError as exception:
        logger.info("--- The selected model was not found, check names")
        logger.debug(exception)
    whisper_model.to(device)

    processor = AutoProcessor.from_pretrained(whisper_model_id)

    if whisper_pipeline_args is None:
        baseline_whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
    else:
        baseline_whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            **whisper_pipeline_args,
        )
    logger.info("--- Baseline Whisper pipeline is created ---")

    return baseline_whisper_pipe


def eval_baseline_model(
    whisper_pipeline, validation_dataset: Dataset
) -> dict[str, float]:
    """Compute WER metrics.

    Args:
        whisper_pipeline (_type_): Whisper baseline pipeline.
        validation_dataset (Dataset):HugginFace dataset used for
        validation purposes.

    Returns:
        dict[str, float]: WER and WER-normalised computed.
    """
    logger.info("--- Launch Baseline Whisper Evaluation ---")

    predictions = []
    references = []

    for _, sample in enumerate(tqdm(validation_dataset)):

        result = whisper_pipeline(
            sample["audio"],
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
        predictions.append(result["text"])
        references.append(sample["transcript"])

    return compute_wer(
        list_predictions=predictions, list_references=references
    )


def eval_checkpoint_model(
    checkpoint_model: torch.nn.Module,
    validation_dataset: Dataset,
    data_collator,
    whisper_processor,
) -> dict[str, int]:
    """Compute WER metrics on Fine-tuned Whisper model.

    Args:
        checkpoint_model (torch.nn.Module): Whisper checkpoint model.
        validation_dataset (Dataset): dataset used to compute metrics
        on unseen data.
        data_collator (_type_): data collator to preprocess data
            before feeding it to the model.
        whisper_processor (_type_): Whisper pretrained processsor to
            encode inputs.

    Returns:
        dict[str, int]: computed metrics
    """

    logger.info("--- Launch Whisper FIne-tuned evaluation ---")

    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )

    predictions = []
    references = []
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=16, collate_fn=data_collator
    )

    for _, sample in enumerate(tqdm(validation_dataloader)):

        with torch.no_grad():
            generated_tokens = (
                checkpoint_model.generate(
                    input_features=sample["input_features"].to("cuda"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = sample["labels"].cpu().numpy()
            labels = np.where(
                labels != -100,
                labels,
                whisper_processor.tokenizer.pad_token_id,
            )
            decoded_preds = whisper_processor.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = whisper_processor.tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

    return compute_wer(
        list_predictions=predictions, list_references=references
    )
