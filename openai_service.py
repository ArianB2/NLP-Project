"""OpenAI connection, streaming calls, and in-memory transcription sessions.

https://developers.openai.com/api/docs/guides/realtime-transcription
Uses the WebSocket API directly; the OpenAI SDK is not required.
"""

import asyncio
import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from websockets.asyncio.client import connect

SAMPLE_RATE = 24_000
FRAME_MS = 20
REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"


@dataclass
class Transcript:
    item_id: str
    text: str = ""
    final: bool = False


class OpenAITranscriptionSession:
    """One connection and its transcripts; use as an async context manager."""

    def __init__(self, *, delay: str = "low", languages: list[str] | None = None):
        load_dotenv(Path(__file__).with_name(".env"))
        self._api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self._api_key:
            raise ValueError("Set a nonempty OPENAI_API_KEY in .env before recording.")
        self.delay = delay
        self.languages = languages
        self.session_id: str | None = None
        self.transcripts: dict[str, Transcript] = {}
        self.item_order: list[str] = []
        self._socket = None
        self._buffered_bytes = 0
        self._pending = 0
        self._drained = asyncio.Event()
        self._drained.set()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def connect(self) -> None:
        """Authenticate and wait until the server accepts the session settings."""
        if self._socket is not None:
            raise RuntimeError("This session is already connected.")
        self._socket = await connect(
            REALTIME_URL,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            open_timeout=15, close_timeout=5, max_size=4 * 1024 * 1024,
        )
        try:
            transcription = {"model": "gpt-live-transcribe", "delay": self.delay}
            if self.languages:
                transcription["languages"] = self.languages
            await self._send({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {"input": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "transcription": transcription,
                        "turn_detection": None,
                    }},
                },
            })
            async with asyncio.timeout(15):
                while True:
                    event = json.loads(await self._socket.recv())
                    self._raise_error(event)
                    if event["type"] == "session.updated":
                        self.session_id = event["session"]["id"]
                        return
        except BaseException:
            await self.close()
            raise

    async def _send(self, event: dict) -> None:
        if self._socket is None:
            raise RuntimeError("Connect the transcription session first.")
        await self._socket.send(json.dumps(event))

    @staticmethod
    def _raise_error(event: dict) -> None:
        if event["type"] in {"error", "conversation.item.input_audio_transcription.failed"}:
            error = event.get("error", {})
            raise RuntimeError(f"OpenAI transcription error: {error.get('message', error)}")

    async def append_audio(self, pcm: bytes) -> None:
        """Send raw mono little-endian PCM16 at 24 kHz, without a WAV header."""
        if not pcm:
            return
        if len(pcm) % 2:
            raise ValueError("PCM16 audio must contain complete two-byte samples.")
        await self._send({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        })
        self._buffered_bytes += len(pcm)

    async def commit(self) -> bool:
        """Finalize a nonempty turn, padding very short turns to 100 ms."""
        if not self._buffered_bytes:
            return False
        minimum_bytes = SAMPLE_RATE * 2 // 10
        if self._buffered_bytes < minimum_bytes:
            await self.append_audio(bytes(minimum_bytes - self._buffered_bytes))
        self._pending += 1
        self._drained.clear()
        await self._send({"type": "input_audio_buffer.commit"})
        self._buffered_bytes = 0
        return True

    async def receive_transcripts(self, on_update: Callable[[Transcript], None]) -> None:
        """Run one receiver alongside the audio sender; report partial/final text."""
        if self._socket is None:
            raise RuntimeError("Connect the transcription session first.")
        async for message in self._socket:
            event = json.loads(message)
            self._raise_error(event)
            kind = event["type"]
            if kind == "input_audio_buffer.committed":
                self.item_order.append(event["item_id"])
            elif kind in {
                "conversation.item.input_audio_transcription.delta",
                "conversation.item.input_audio_transcription.completed",
            }:
                item_id = event["item_id"]
                transcript = self.transcripts.setdefault(item_id, Transcript(item_id))
                if kind.endswith(".delta"):
                    transcript.text += event.get("delta", "")
                elif not transcript.final:
                    transcript.text = event["transcript"]
                    transcript.final = True
                    self._pending -= 1
                    if self._pending == 0:
                        self._drained.set()
                on_update(transcript)
        raise ConnectionError("OpenAI closed the transcription connection.")

    async def wait_for_transcripts(self, timeout: float = 15) -> None:
        """Keep receiving until all committed turns have final text."""
        await asyncio.wait_for(self._drained.wait(), timeout)

    @property
    def text(self) -> str:
        """Final text in audio commit order, regardless of completion order."""
        return "\n".join(
            self.transcripts[item_id].text for item_id in self.item_order
            if item_id in self.transcripts and self.transcripts[item_id].final
        )

    async def close(self) -> None:
        if self._socket is not None:
            await self._socket.close()
            self._socket = None
