from faster_whisper import WhisperModel
import sys
import logging

from typing import Union, List

logger = logging.getLogger(__name__)


class FasterWhisperASR:
    sep = ""

    def __init__(
        self,
        model_size: str = None,
        device_index: Union[List[int], int] = 0,
        device: str = "cpu",
        compute_type: str = "float32",
        extra_transcribe_args: dict[str, Union[str, int]] = {},
        logfile: str = sys.stderr,
    ) -> None:
        """Construct basic FastWhisper implementation.

        Args:
            model_size (str, optional): model to use. Defaults to None.
            device_index (Union[List[int], int], optional):
                List of gpu index to use. Defaults to 0.
            device (str, optional): whether to use "cpu" or "gpu".
                Defaults to "cpu".
            compute_type (str, optional): dtypes for operations
                during computation. Defaults to "float32".
            logfile (str, optional): Where to output transcriptions.
                Defaults to sys.stderr.
        """

        logger.info(f"Model used : {model_size} -- Device : {device}")
        self.model_size = model_size
        self.logfile = logfile

        self.extra_transcribe_args = extra_transcribe_args

        self.model = self._load_model(
            compute_type=compute_type, device=device, device_index=device_index
        )

    def _load_model(
        self,
        compute_type: str,
        device: str,
        device_index: Union[List[int], int],
    ):
        """Load FasterWhisper model.

        Args:
            compute_type (str): computation dtype.
            device (str): device on which the model will be attached.
            device_index (Union[List[int], int]): list of gpu indexes.

        Returns:
            _type_: _description_
        """

        faster_model = WhisperModel(
            model_size_or_path=self.model_size,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
        )
        return faster_model

    def transcribe(
        self,
        audio_file: str,
        language_selected: str = "fr",
        beam_size: int = 5,
        init_prompt: str = "",
    ) -> List[str]:
        """Transcribe single piece of audio.

        Args:
            audio_file (str): Audio file to transcribe.
            language_selected (str, optional): Selected language.
                Defaults to "fr".
            beam_size (int, optional): beam_size to use for decoding.
                Defaults to 5.
            init_prompt (str, optional): optional text string
                for the first window. Defaults to "".

        Returns:
            List[str]: list of transcriptions of each segment.
        """

        segments, _ = self.model.transcribe(
            audio_file,
            language=language_selected,
            initial_prompt=init_prompt,
            beam_size=beam_size,
            word_timestamps=True,
            condition_on_previous_text=True,
            task="transcribe",
            **self.extra_transcribe_args,
        )

        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]
