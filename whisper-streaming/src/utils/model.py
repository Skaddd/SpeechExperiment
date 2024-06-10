import logging
import sys
from typing import Any, List, NamedTuple, Union

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class Word(NamedTuple):
    start: float
    end: float
    word: str


class SWhisperModel:

    def __init__(
        self,
        language: str,
        model_size: str = None,
        device_index: Union[List[int], int] = 0,
        device: str = "cpu",
        compute_type: str = "float32",
        logfile: str = sys.stderr,
    ) -> None:
        """Construct basic FastWhisper implementation.

        Args:
            language (str): Which language to use during transcription.
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
        self.output_destination = logfile
        self.language = language

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
            SWhisperModel: Modified WHisperModel.
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
        beam_size: int = 5,
        init_prompt: str = "",
    ) -> List[str]:
        """Transcribe single piece of audio.

        Args:
            audio_file (str): Audio file to transcribe.
            beam_size (int, optional): beam_size to use for decoding.
                Defaults to 5.
            init_prompt (str, optional): optional text string
                for the first window. Defaults to "".

        Returns:
            List[str]: list of transcriptions of each segment.
        """

        segments, _ = self.model.transcribe(
            audio_file,
            language=self.language,
            initial_prompt=init_prompt,
            beam_size=beam_size,
            word_timestamps=True,
            condition_on_previous_text=True,
            task="transcribe",
            vad_filter=True,
        )

        return list(segments)

    def timestamps_words(self, segments: list[NamedTuple]) -> list[NamedTuple]:
        """Generate standardized segments to word timestamps.

        Args:
            segments (list[NamedTuple]): Segments from faster-whisper implem.

        Returns:
            list[NamedTuple]: list of Words.
        """
        words_output = []
        for segment in segments:
            for word in segment.words:
                # not stripping the spaces -- should not be merged with them!
                words_output.append(
                    Word(start=word.start, end=word.end, word=word.word)
                )
        return words_output

    def segments_end_timestamps(
        self, transcripted_words_list: list[NamedTuple]
    ) -> list[NamedTuple]:
        """Extract ending timestamps for a sentence.

        Args:
            transcripted_words_list (list[NamedTuple]): Transcripted words.

        Returns:
            list[NamedTuple]: Ending timestamps of a given setence.
        """
        return [s.end for s in transcripted_words_list]


class OnlineWhisperPipeline:
    """OnlineWhisperPipeline processes audio chunks in real-time.

    This class allows for the continuous ingestion of audio chunks,
    buffers them, and uses a Whisper model to transcribe the buffered audio.
    It manages buffering, prompt generation for context, and handles trimming
    of the audio buffer based on configurable parameters.

    Attributes:
        whisper_model (SWhisperModel): An instance of the Whisper model
            used for transcription.
        output_destination (Any): Destination for outputting transcription
            results, typically a file or stderr.
        buffer_trimming_sec (int): Duration in seconds to determine
            when to trim the audio buffer.
        audio_buffer (np.array): Buffer to hold the incoming audio chunks.
        buffer_time_offset (float): Time offset for the current audio buffer.
        transcript_buffer (BufferTracker): Buffer to track transcriptions.
        transcription_results (List): List to store the transcribed results.
    """

    def __init__(
        self,
        whisper_model,
        buffer_trimming_sec: int = 15,
        output_destination: Any = sys.stderr,
    ):
        """
        Constructor of OnlineWhisperPipeline.

        Args:
            whisper_model (SWhisperModel): Faster-whisper model.
            buffer_trimming_sec (int, optional): Buffer trimming duration
                in seconds. Defaults to 15.
            output_destination (Any, optional): Output destination for
                transcription results. Defaults to sys.stderr.
        """
        self.whisper_model = whisper_model
        self.output_destination = output_destination
        self.buffer_trimming_sec = buffer_trimming_sec
        self.initialize_buffers()

    def initialize_buffers(self):
        """Initialize the buffers for audio and transcription."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0
        self.transcript_buffer = BufferTracker(logfile=self.output_destination)
        self.transcription_results = []

    def insert_audio_chunk(self, audio_chunk):
        """Append an audio chunk to the audio buffer.

        Args:
            audio_chunk (np.array): Incoming audio chunk.
        """
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)

    def generate_initial_prompt(self, max_characters: int = 200) -> str:
        """Generate a prompt for the audio chunk.

        This function uses the `init_prompt` parameter
        of the faster-whisper implementation to provide context
        to the model for transcription.

        Args:
            max_characters (int, optional): Maximum number of characters
              the prompt can have. Defaults to 200.
        Returns:
            str: Prompt for the next audio chunk.
        """
        if not self.transcription_results:
            return ""

        prompt_length = 0
        prompt_words = []

        for word in reversed(self.transcription_results):
            if word.end > self.buffer_time_offset:
                break

            if prompt_length + len(word.word) <= max_characters:
                prompt_words.append(word.word)
                prompt_length += len(word.word)
            else:
                break

        return "".join(prompt_words)

    def process_iteration(self):
        """Process the current audio buffer.

        This function aims at transcribing the
        audio_buffer which contains all the audios chunks accessible so far.
        Prompt of the previous transcription inference is also passed.
        Once results are obtained, overlapping words must be removed.
        Returns:
            tuple: A final Word.
        """
        prompt = self.generate_initial_prompt()
        chunk_transcription = self.whisper_model.transcribe(
            self.audio_buffer, init_prompt=prompt
        )

        # Transform to [(start, end, "word1"), ...]
        transcription_word_list = self.whisper_model.timestamps_words(
            chunk_transcription
        )

        self.transcript_buffer.insert(
            transcription_word_list, self.buffer_time_offset
        )
        committed_words = self.transcript_buffer.flush()
        self.transcription_results.extend(committed_words)

        if len(self.audio_buffer) / SAMPLING_RATE > self.buffer_trimming_sec:
            self.complete_audio_segment(transcription_word_list)

        return self.format_transcription(committed_words)

    def complete_audio_segment(
        self, transcription_word_list: list[NamedTuple]
    ):
        """Complete the transcription for the audio segment.

        Args:
            transcription_word_list (list): List of transcribed
            words with timestamps.
        """
        if not self.transcription_results:
            return

        segment_end_timestamps = self.whisper_model.segments_end_timestamps(
            transcription_word_list
        )
        last_transcription_end = self.transcription_results[-1].end

        if len(segment_end_timestamps) > 1:
            segment_end_time = (
                segment_end_timestamps[-2] + self.buffer_time_offset
            )
            while (
                len(segment_end_timestamps) > 2
                and segment_end_time > last_transcription_end
            ):
                segment_end_timestamps.pop(-1)
                segment_end_time = (
                    segment_end_timestamps[-2] + self.buffer_time_offset
                )

            if segment_end_time <= last_transcription_end:
                logger.debug(f"Segment chunked at {segment_end_time:.2f}")
                self.trim_buffers_at(segment_end_time)

    def trim_buffers_at(self, time):
        """
        Trim the hypothesis and audio buffer at the specified time.

        Args:
            time (float): The time to trim the buffers.
        """
        self.transcript_buffer.pop_committed(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[
            int(cut_seconds * SAMPLING_RATE):
        ]
        self.buffer_time_offset = time

    def finish(self):
        """Flush the incomplete text when the processing ends.

        Returns:
            tuple: The same format as `process_iteration()`.
        """
        incomplete_words = self.transcript_buffer.complete()
        final_transcription = self.format_transcription(incomplete_words)
        logger.debug(
            f"Last non-committed transcription: {final_transcription}"
        )
        return final_transcription

    def format_transcription(
        self, words: list[NamedTuple], separator: str = ""
    ):
        """Format the transcribed words into a single sequence.

        Args:
            words (list[NamedTuple]): List of Words.
            separator (str, optional): Separator for joining words.
                Defaults to "".

        Returns:
            NamedTuple: Word.
        """
        if not words:
            return None, None, ""

        transcription_text = separator.join(word.word for word in words)
        start_time = words[0].start + self.buffer_time_offset
        end_time = words[-1].end + self.buffer_time_offset

        return Word(start=start_time, end=end_time, word=transcription_text)


class BufferTracker:
    """
    Tracks transcription buffer and handles insertion, flushing,
    and committing of words.

    Parameters
    ----------
    logfile : file object, optional
        A file object for logging (default is sys.stderr).
    """

    def __init__(self, logfile=sys.stderr):
        self.transcription_results_buffer = []
        self.current_buffer = []
        self.new_words_buffer = []

        self.last_committed_time = 0
        self.last_committed_word = None

        self.logfile = logfile

    def insert(
        self,
        transcripted_words_list: List[Word],
        offset: float,
        n_grams_range: int = 5,
    ) -> None:
        """
        Insert new transcribed words into the buffer.

        This method compares the new transcribed words with
        the existing transcription results in the buffer
        and inserts only the words that extend the committed buffer.
        The new tail is added to the new words buffer.

        Parameters
        ----------
        transcripted_words_list : list of Word
            List of recently transcribed words.
        offset : float
            Time offset to adjust the start and end times of the words.
        n_grams_range : int, optional
            The range of n-grams to consider for matching (default is 5).

        Returns
        -------
        None
        """
        # Adjust start and end times of the words by the offset
        recently_transcribed_words = [
            Word(
                start=word.start + offset,
                end=word.end + offset,
                word=word.word,
            )
            for word in transcripted_words_list
        ]

        # Extract expected unseen words
        self.new_words_buffer = [
            word
            for word in recently_transcribed_words
            if word.start > self.last_committed_time - 0.1
        ]

        if len(self.new_words_buffer) > 0:
            first_word = self.new_words_buffer[0]
            if abs(first_word.start - self.last_committed_time) < 1:
                if self.transcription_results_buffer:
                    length_prev_buffer = len(self.transcription_results_buffer)
                    length_new_buffer = len(self.new_words_buffer)
                    sequence_length = (
                        min(
                            min(
                                length_prev_buffer,
                                length_new_buffer,
                            ),
                            n_grams_range,
                        )
                        + 1
                    )
                    for i in range(1, sequence_length):
                        previous_sequence = " ".join(
                            [
                                self.transcription_results_buffer[-j].word
                                for j in range(1, i + 1)
                            ][::-1]
                        )
                        new_sequence = " ".join(
                            self.new_words_buffer[j - 1].word
                            for j in range(1, i + 1)
                        )
                        if previous_sequence == new_sequence:
                            for j in range(i):
                                self.new_words_buffer.pop(0)
                            break

    def flush(self) -> List[Word]:
        """Flushes the new words buffer and returns the committed chunk.

        The committed chunk is the longest common
          prefix of the last two inserts.

        Returns
        -------
        list of Word
            The committed chunk of words.
        """
        committed_chunk = []
        while self.new_words_buffer:
            first_word = self.new_words_buffer[0]

            if len(self.current_buffer) == 0:
                break

            if first_word.word == self.current_buffer[0].word:
                committed_chunk.append(first_word)
                self.last_committed_word = first_word.word
                self.last_committed_time = first_word.end
                self.current_buffer.pop(0)
                self.new_words_buffer.pop(0)
            else:
                break

        self.current_buffer = self.new_words_buffer
        self.new_words_buffer = []
        self.transcription_results_buffer.extend(committed_chunk)
        return committed_chunk

    def pop_committed(self, time: float) -> None:
        """
        Remove committed words from the buffer that are before the given time.

        Parameters
        ----------
        time : float
            The time threshold for removing committed words.

        Returns
        -------
        None
        """
        while (
            self.transcription_results_buffer
            and self.transcription_results_buffer[0].end <= time
        ):
            self.transcription_results_buffer.pop(0)

    def complete(self) -> List[Word]:
        """
        Get the current buffer of words.

        Returns
        -------
        list of Word
            The current buffer of words.
        """
        return self.current_buffer
