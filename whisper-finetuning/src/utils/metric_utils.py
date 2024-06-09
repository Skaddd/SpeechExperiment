import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import logging


logger = logging.getLogger(__name__)


def compute_wer(
    list_predictions: list[str], list_references: list[str]
) -> float:
    """Compute both WER metric and WER normalised.

    This function follows Whisper's paper
    about creating a new metric called normalised-WER.
    The goal is to evalute the syntaxic correctness of any
    STT models by working on normalised text without punctuation
    and specific characters.
    Args:
        list_predictions (list[str]): list of predicted transcriptions.
        list_references (list[str]): list of ground truth transcriptions.

    Returns:
        float: WER score when comparing predicition results with
        GT values.
    """
    logger.info("TEST")
    normalizer = BasicTextNormalizer()

    error_message = "References and predicitions inputs length miss match"
    assert len(list_predictions) == len(list_references), error_message

    wer_metric = evaluate.load("wer")

    raw_wer = 100 * wer_metric.compute(
        predictions=list_predictions, references=list_references
    )
    normalised_wer = 100 * wer_metric.compute(
        predictions=[
            normalizer(predictions).strip() for predictions in list_predictions
        ],
        references=[
            normalizer(reference).strip() for reference in list_references
        ],
    )

    return {"raw_wer_score": raw_wer, "normalised_wer_score": normalised_wer}
