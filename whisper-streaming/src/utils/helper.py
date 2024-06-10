def format_milliseconds(seconds_input: float) -> str:
    """Format seconds to pretty printing

    Args:
        seconds_input (float): Number of seconds.

    Returns:
        str: pretty printing seconds
    """
    if isinstance(seconds_input, str):
        seconds_input = float(seconds_input)

    minutes, seconds = divmod(seconds_input, 60)
    milliseconds = (seconds_input - int(seconds_input)) * 1000

    return f"{int(minutes):02}:{int(seconds):02}:{int(milliseconds):03}"


def output_transcriptions(
    audio_chunk_transcription: tuple,
    time_tracker_from_start: float,
    logfile: str,
) -> None:
    """Format Transcription Results to pretty printing.

    Args:
        audio_chunk_transcription (tuple): Tuple containing
        timestamps and segment transcribed.
        time_tracker_from_start (float): Time tracker to compute
        time elapsed from the begining of the transcription.
        logfile (str): logfile to output the results.
    """

    if audio_chunk_transcription[0] is not None:
        print(
            "%s [%s - %s] -- %s"
            % (
                format_milliseconds(time_tracker_from_start),
                format_milliseconds(audio_chunk_transcription.start),
                format_milliseconds(audio_chunk_transcription.end),
                audio_chunk_transcription.word,
            ),
            file=logfile,
            flush=True,
        )
