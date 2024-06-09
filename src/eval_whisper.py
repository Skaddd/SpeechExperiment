import logging

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from utils.helpers import setup_logging
from utils.metric_utils import compute_wer

logger = logging.getLogger(__name__)
setup_logging()


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


if __name__ == "__main__":
    validation_dataset = load_dataset(
        "tobiolatunji/afrispeech-200", "isizulu", split="test"
    )

    whisper_pipeline = load_baseline_whisper_model("openai/whisper-tiny")

    eval_baseline_model(whisper_pipeline, validation_dataset)
