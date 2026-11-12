"""Offline protocol and audio-pipeline checks; no microphone or API calls."""

import asyncio
import base64
import json
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from openai_service import OpenAITranscriptionSession
from record_STT import stream_microphone


class FakeSocket:
    def __init__(self):
        self.events = asyncio.Queue()
        self.sent = []
        self.close = AsyncMock()

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        return json.dumps(await self.events.get())

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.recv()


class STTTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self):
        with patch("openai_service.load_dotenv"), patch.dict(
            "os.environ", {"OPENAI_API_KEY": "offline-test"}
        ):
            return OpenAITranscriptionSession()

    async def test_session_configuration_and_cleanup(self):
        socket = FakeSocket()
        socket.events.put_nowait({"type": "session.created", "session": {"id": "s1"}})
        socket.events.put_nowait({"type": "session.updated", "session": {"id": "s1"}})
        with patch("openai_service.connect", AsyncMock(return_value=socket)) as connect:
            async with self.make_session() as session:
                self.assertEqual(session.session_id, "s1")
                self.assertIn("intent=transcription", connect.call_args.args[0])
                config = socket.sent[0]["session"]["audio"]["input"]
                self.assertEqual(config["format"], {"type": "audio/pcm", "rate": 24000})
                self.assertEqual(config["transcription"]["model"], "gpt-live-transcribe")
                self.assertIsNone(config["turn_detection"])
        socket.close.assert_awaited_once()

    async def test_audio_and_out_of_order_transcripts(self):
        session = self.make_session()
        socket = session._socket = FakeSocket()
        self.assertFalse(await session.commit())
        for _ in range(2):
            await session.append_audio(bytes(960))
            await session.commit()
        # Short turns are padded to the minimum buffer before committing.
        self.assertEqual(len(base64.b64decode(socket.sent[1]["audio"])), 3840)
        updates = []
        receiver = asyncio.create_task(session.receive_transcripts(
            lambda item: updates.append((item.item_id, item.text, item.final))
        ))
        for item in ["a", "b"]:
            socket.events.put_nowait({"type": "input_audio_buffer.committed", "item_id": item})
        socket.events.put_nowait({
            "type": "conversation.item.input_audio_transcription.delta", "item_id": "b", "delta": "Sec",
        })
        for item, text in [("b", "Second"), ("a", "First")]:
            socket.events.put_nowait({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item, "transcript": text,
            })
        try:
            await session.wait_for_transcripts(timeout=1)
            self.assertEqual(session.text, "First\nSecond")
            self.assertIn(("b", "Sec", False), updates)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    async def test_api_error_is_propagated(self):
        session = self.make_session()
        socket = session._socket = FakeSocket()
        socket.events.put_nowait({"type": "error", "error": {"message": "Rejected"}})
        with self.assertRaisesRegex(RuntimeError, "Rejected"):
            await session.receive_transcripts(lambda item: None)

    async def test_silence_preroll_and_stop_flush(self):
        from types import SimpleNamespace
        mic = SimpleNamespace(frames=asyncio.Queue(), error=None, stop=lambda: None)
        session = SimpleNamespace(
            append_audio=AsyncMock(), commit=AsyncMock(), wait_for_transcripts=AsyncMock(),
        )
        silence = bytes(960)
        speech = np.full(480, 2000, dtype="<i2").tobytes()
        # Initial silence is bounded to the pre-roll. Stop during a second turn.
        for frame in [silence] * 20 + [speech] + [silence] * 25 + [speech]:
            mic.frames.put_nowait(frame)
        stop = asyncio.Event()
        stop.set()
        await stream_microphone(mic, session, stop)
        self.assertEqual(session.append_audio.await_count, 36)
        self.assertEqual(session.commit.await_count, 2)
        session.wait_for_transcripts.assert_awaited_once()

    async def test_missing_key_fails_before_connecting(self):
        with patch("openai_service.load_dotenv"), patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "nonempty OPENAI_API_KEY"):
                OpenAITranscriptionSession()


if __name__ == "__main__":
    unittest.main()
