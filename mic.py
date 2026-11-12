"""macOS microphone capture through PortAudio's native Core Audio backend."""

import asyncio
import sys

import sounddevice as sd


class Microphone:
    def __init__(self, *, sample_rate: int = 24_000, frame_ms: int = 20, device=None):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.device = device
        self.frames: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self.error: str | None = None
        self._stream = None

    def start(self) -> None:
        """Start capture; call inside the running asyncio event loop."""
        if sys.platform != "darwin":
            raise RuntimeError("Microphone capture currently supports macOS only.")
        if self._stream is not None:
            return
        loop = asyncio.get_running_loop()
        self.error = None

        def enqueue(pcm, status):
            if status:
                self.error = f"Microphone capture failed: {status}"
            elif self.frames.full():
                self.error = "Audio queue filled; the connection cannot keep up."
            else:
                self.frames.put_nowait(pcm)

        def callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(enqueue, bytes(indata), str(status))

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.sample_rate * self.frame_ms // 1000,
                channels=1, dtype="int16", device=self.device,
                latency="low", callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            self.stop()
            raise RuntimeError(
                "Cannot start the Mac microphone. Check the input device and enable "
                "microphone access for your terminal or IDE in System Settings > "
                "Privacy & Security > Microphone."
            ) from exc

    def stop(self) -> None:
        """Stop and release the microphone; safe to call repeatedly."""
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
