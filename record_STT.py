"""Run: env/bin/python record_STT.py — speak, then press Ctrl+C to finish."""

import argparse
import asyncio
from collections import deque
import signal
import sys

import numpy as np
import sounddevice as sd

from mic import Microphone
from openai_service import FRAME_MS, OpenAITranscriptionSession, Transcript


async def stream_microphone(mic, session, stop, *, threshold=500, silence_ms=500):
    """Stream speech with 200 ms pre-roll; commit after a short silence.

    Simple energy-based speech detection: adjust --threshold for your room.
    Higher values ignore more background noise. Cap turns at 30 seconds.
    """
    preroll = deque(maxlen=200 // FRAME_MS)
    active = False
    quiet_ms = turn_ms = 0
    try:
        while not stop.is_set() or not mic.frames.empty():
            if mic.error:
                raise RuntimeError(mic.error)
            try:
                pcm = await asyncio.wait_for(mic.frames.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
            speech = np.sqrt(np.mean(samples * samples)) >= threshold
            if not active:
                preroll.append(pcm)
                if not speech:
                    continue
                active = True
                turn_ms = len(preroll) * FRAME_MS
                for frame in preroll:
                    await session.append_audio(frame)
                preroll.clear()
            else:
                await session.append_audio(pcm)
                turn_ms += FRAME_MS
            quiet_ms = 0 if speech else quiet_ms + FRAME_MS
            if quiet_ms >= silence_ms or turn_ms >= 30_000:
                await session.commit()
                active = False
                quiet_ms = turn_ms = 0
        await session.commit()
        await session.wait_for_transcripts()
    finally:
        mic.stop()


def display_transcript(transcript: Transcript) -> None:
    # Label items because separate turns can complete out of order.
    state = "final" if transcript.final else "partial"
    print(f"[{state} {transcript.item_id}] {transcript.text}", flush=True)


async def record(args) -> str:
    """Connect first, record until Ctrl+C, then wait for the final transcript."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    mic = Microphone(device=args.device)

    def stop_recording():
        mic.stop()
        stop.set()

    async with OpenAITranscriptionSession(delay=args.delay, languages=args.languages) as session:
        tasks = []
        loop.add_signal_handler(signal.SIGINT, stop_recording)
        try:
            mic.start()
            print("Microphone on. Speak to see live transcription below.\n"
                  "Press Ctrl+C to stop and show the full transcript.\n", flush=True)
            receiver = asyncio.create_task(session.receive_transcripts(display_transcript))
            sender = asyncio.create_task(stream_microphone(
                mic, session, stop, threshold=args.threshold, silence_ms=args.silence_ms,
            ))
            tasks = [receiver, sender]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            return session.text
        finally:
            mic.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            loop.remove_signal_handler(signal.SIGINT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live OpenAI STT from the Mac microphone.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, help="Input device index; default: system microphone")
    parser.add_argument("--languages", nargs="+", help="Expected language codes, e.g. en fr")
    parser.add_argument("--delay", choices=["minimal", "low", "medium", "high", "xhigh"], default="low")
    parser.add_argument("--threshold", type=int, default=500, help="Speech RMS threshold, 1–32767")
    parser.add_argument("--silence-ms", type=int, default=500, help="Silence before finalizing a turn")
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    if not 1 <= args.threshold <= 32767 or args.silence_ms < FRAME_MS:
        parser.error("threshold must be 1–32767 and silence-ms must be at least 20")
    try:
        text = asyncio.run(record(args))
        if text:
            print(f"\nSession transcript:\n{text}")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"STT stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
