import argparse
import glob
import logging
import os
import sys
import time
from functools import lru_cache
from typing import Any

import librosa
import numpy as np

from .utils.helper import output_transcriptions
from .utils.model import OnlineWhisperPipeline, SWhisperModel

logger = logging.getLogger(__name__)


@lru_cache
def load_audio(audio_file: str) -> np.array:
    """Load audio file in cache.

    Args:
        audio_file (str): file path to audio file.

    Returns:
        np.array: Audio file transformed and downsampled.
    """
    a, _ = librosa.load(audio_file, sr=16000, dtype=np.float32)
    return a


def load_audio_chunk(audio_file: str, beg: float, end: float) -> np.array:
    """Load piece of audio file.

    This function select a peice of a given audio file.
    Args:
        audio_file (str): file path to audio file.
        beg (float): Starting time in seconds.
        end (float): Ending time in seconds.

    Returns:
        np.array: Chunk of the audio array.
    """
    audio = load_audio(audio_file)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


def parsing_args_script() -> None:

    parser = argparse.ArgumentParser(
        description="Description of parameters"
        + "-- Multiprocessing is disabled by default"
    )
    required_group = parser.add_argument_group("Required parameters")
    required_group.add_argument(
        *["--audio_file", "-f"],
        help="Audio file to transcribe in streaming way",
    )
    required_group.add_argument(
        *["--audio-folder", "-folder"],
        help="Folder containing audio files in mp3/wav format",
    )
    parser.add_argument(
        *["--model_size", "--model"],
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model selector",
    )

    parser.add_argument(
        *["--min-chunk", "-chunk"],
        type=float,
        default=1.5,
        help="Min audio chunk size in seconds.",
    )

    parser.add_argument(
        *["--language", "-l"],
        type=str,
        default="en",
        help="Language spoken in audios",
    )
    parser.add_argument(
        *["--device", "-dev"],
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Define on which resources the transcription should be launched",
    )
    parser.add_argument(
        *["--compute-type", "-compute"],
        default="float32",
        help="selecting compute precision during inference",
    )

    return parser.parse_args()


def create_whisper_streamer(args: dict[Any, Any]) -> OnlineWhisperPipeline:
    """Instantiate a WhisperStreamer Object.

    Args:
        args (dict[Any, Any]): argparser.

    Returns:
        OnlineWhisperPipeline: Raw online Streamer.
    """

    swhisper_model = SWhisperModel(
        language=args.language,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    streaming_whisper = OnlineWhisperPipeline(swhisper_model)
    logger.info(
        f"Loading Whisper {args.model_size} model for {args.language}..."
    )

    return streaming_whisper


def warm_up_whisper(
    streaming_whisper: OnlineWhisperPipeline, arbitrary_audio_file: str
) -> None:
    """Warm_up Whisper model.

    This function aims at warming-up the faster whisper model
    simply because the first inference takes longer.
    Args:
        streaming_whisper (OnlineWhisperPipeline): Streaming Whipser.
        arbitrary_audio_file (str): any audio file.
    """

    _ = load_audio_chunk(arbitrary_audio_file, 0, 1)

    streaming_whisper.whisper_model.transcribe(_)


def online_transcription(
    audio_path: str,
    min_chunk: float,
    streaming_whisper: OnlineWhisperPipeline,
    logfile: Any,
):
    """Launch Online transcription on a given file

    Args:
        audio_path (str): file path to the audio file.
        min_chunk (float): Min chunk size to consider.
        streaming_whisper (OnlineWhisperPipeline): Whisper streamer
            model.
        logfile (Any): Where the output should be displayed.

    Returns:
        str: final transcription merged.
    """
    # Reset buffers for a new file
    streaming_whisper.initialize_buffers()

    duration = len(load_audio(audio_path)) / 16000

    starting_time = 0
    time_processed = 0
    start = time.time() - starting_time

    while True:
        time_tracker_from_start = time.time() - start
        # Creating a small chunk of time to be then passed to whisper.
        if time_tracker_from_start < time_processed + min_chunk:
            time.sleep(time_processed + min_chunk - time_tracker_from_start)
        time_processed = time.time() - start

        # Audio chunk is computed
        # Starting time becomes time processed
        audio_chunk = load_audio_chunk(
            audio_path, starting_time, time_processed
        )
        starting_time = time_processed

        streaming_whisper.insert_audio_chunk(audio_chunk)

        try:
            new_transcribed_words = streaming_whisper.process_iteration()
        except AssertionError as e:
            logger.debug(f"assertion error: {e}")
            pass
        else:
            output_transcriptions(
                audio_chunk_transcription=new_transcribed_words,
                time_tracker_from_start=time_tracker_from_start,
                logfile=logfile,
            )
        if time_processed >= duration:
            break

    new_transcribed_words = streaming_whisper.finish()
    output_transcriptions(
        audio_chunk_transcription=new_transcribed_words,
        time_tracker_from_start=time_tracker_from_start,
        logfile=logfile,
    )

    return {
        audio_path: "".join(
            [word.word for word in streaming_whisper.transcription_results]
        )
    }


def multi_online_transcription(
    audio_folder: str, min_chunk: float, streaming_whisper, logfile
) -> dict[str, str]:
    """Launch streaming trancription for multiple files.

    Args:
        audio_folder (str): Folder containing audio files.
        min_chunk (float): Min chunk size to consider.
        streaming_whisper (OnlineWhisperPipeline): Whisper streamer
            model.
        logfile (Any): Where the output should be displayed.

    Returns:
        dict[str, str]: final transcription merged.
    """

    transcription = {}

    for audio_file in glob.glob(os.path.join(audio_folder, "*")):
        transcription.update(
            online_transcription(
                audio_path=audio_file,
                min_chunk=min_chunk,
                streaming_whisper=streaming_whisper,
                logfile=logfile,
            )
        )

    return transcription


def apply_transcription_streaming():

    args = parsing_args_script()
    # reset to store stderr to different file stream, e.g. open(os.devnull,"w")
    logfile = sys.stderr

    streaming_whisper = create_whisper_streamer(args)

    if args.audio_file:
        warm_up_whisper(
            streaming_whisper=streaming_whisper,
            arbitrary_audio_file=args.audio_file,
        )

        transcription = online_transcription(
            audio_path=args.audio_file,
            min_chunk=args.min_chunk,
            streaming_whisper=streaming_whisper,
            logfile=logfile,
        )
    else:
        transcription = multi_online_transcription(
            audio_folder=args.audio_folder,
            min_chunk=args.min_chunk,
            streaming_whisper=streaming_whisper,
            logfile=logfile,
        )
