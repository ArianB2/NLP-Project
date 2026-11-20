"""Offline hostility-monitor checks; no microphone, model download or API calls."""

import asyncio
import contextlib
import io
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hostility import HostilityMonitor, level_for, split_turn
from openai_service import Transcript
import record_STT


class FakeScorer:
    def __init__(self, scores=None, delay=0.0, fail_on=()):
        self.scores = scores or {}
        self.delay = delay
        self.fail_on = set(fail_on)
        self.calls = []
        self.loaded = False

    def load(self):
        self.loaded = True

    def score(self, text, context):
        self.calls.append((text, [turn.text for turn in context]))
        if self.delay:
            time.sleep(self.delay)
        if text in self.fail_on:
            raise ValueError("bad turn")
        return self.scores.get(text, 0.0)


def make_session(*turns):
    """Build session state from (item_id, text, final) tuples in spoken order."""
    return SimpleNamespace(
        item_order=[item_id for item_id, _, _ in turns],
        transcripts={item_id: Transcript(item_id, text, final) for item_id, text, final in turns},
    )


class HelperTests(unittest.TestCase):
    def test_threshold_boundaries(self):
        self.assertEqual(level_for(0.29, 0.3, 0.7), "ignore")
        self.assertEqual(level_for(0.3, 0.3, 0.7), "check")
        self.assertEqual(level_for(0.69, 0.3, 0.7), "check")
        self.assertEqual(level_for(0.7, 0.3, 0.7), "hostile")

    def test_split_turn(self):
        self.assertEqual(split_turn("   "), [])
        self.assertEqual(split_turn("One. Two."), ["One. Two."])
        first, second = " ".join(["word"] * 15) + ".", " ".join(["more"] * 15) + "?"
        self.assertEqual(split_turn(f"{first} {second}"), [first, second])


class MonitorTests(unittest.IsolatedAsyncioTestCase):
    async def run_monitor(self, monitor, session):
        worker = asyncio.create_task(monitor.run())
        try:
            await monitor.finish(session, timeout=2)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    async def test_out_of_order_finals_are_scored_in_spoken_order(self):
        scorer = FakeScorer()
        monitor = HostilityMonitor(scorer, on_decision=lambda decision: None)
        session = make_session(("a", "first", False), ("b", "second", True))
        monitor.release(session)
        self.assertTrue(monitor._queue.empty())
        session.transcripts["a"] = Transcript("a", "first", True)
        monitor.release(session)
        await self.run_monitor(monitor, session)
        self.assertEqual([text for text, _ in scorer.calls], ["first", "second"])
        self.assertEqual(scorer.calls[1][1], ["first"])

    async def test_context_is_capped_and_only_contains_earlier_turns(self):
        scorer = FakeScorer()
        monitor = HostilityMonitor(scorer, on_decision=lambda decision: None)
        turns = [(str(n), f"turn {n}", True) for n in range(8)]
        await self.run_monitor(monitor, make_session(*turns))
        self.assertEqual(scorer.calls[0][1], [])
        self.assertEqual(scorer.calls[7][1], [f"turn {n}" for n in range(2, 7)])

    async def test_slow_scorer_does_not_block_event_loop(self):
        monitor = HostilityMonitor(FakeScorer(delay=0.2), on_decision=lambda decision: None)
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        await self.run_monitor(monitor, make_session(("a", "slow turn", True)))
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)
        self.assertGreaterEqual(ticks, 5)

    async def test_scorer_error_does_not_stop_later_turns(self):
        scorer = FakeScorer(scores={"after": 0.9}, fail_on={"broken"})
        monitor = HostilityMonitor(scorer, on_decision=lambda decision: None)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            await self.run_monitor(monitor, make_session(("a", "broken", True), ("b", "after", True)))
        self.assertIn("skipped a turn", errors.getvalue())
        self.assertEqual(monitor.errors, 1)
        self.assertEqual([d.level for d in monitor.decisions], ["hostile"])

    async def test_finish_skips_blank_and_unfinished_turns(self):
        scorer = FakeScorer()
        monitor = HostilityMonitor(scorer, on_decision=lambda decision: None)
        session = make_session(("a", "   ", True), ("b", "never final", False), ("c", "done", True))
        await self.run_monitor(monitor, session)
        self.assertEqual([text for text, _ in scorer.calls], ["done"])


class FakeSession:
    instance = None

    def __init__(self, **kwargs):
        self.item_order = ["a", "b"]
        self.transcripts = {}
        self.delivered = asyncio.Event()
        FakeSession.instance = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def receive_transcripts(self, on_update):
        for item_id, text in [("a", "you are useless"), ("b", "nice round")]:
            transcript = self.transcripts.setdefault(item_id, Transcript(item_id, text, True))
            on_update(transcript)
        self.delivered.set()
        await asyncio.Event().wait()

    @property
    def text(self):
        return "\n".join(self.transcripts[item_id].text for item_id in self.item_order)


async def fake_stream(mic, session, stop, **kwargs):
    await session.delivered.wait()


class RecordWiringTests(unittest.IsolatedAsyncioTestCase):
    def args(self, **overrides):
        values = dict(device=None, delay="low", languages=None, threshold=500, silence_ms=500,
                      no_classify=False, check_threshold=0.3, flag_threshold=0.7)
        return SimpleNamespace(**{**values, **overrides})

    async def run_record(self, scorer, **overrides):
        fake_mic = SimpleNamespace(start=lambda: None, stop=lambda: None)
        output = io.StringIO()
        with patch.object(record_STT, "Microphone", return_value=fake_mic), \
                patch.object(record_STT, "OpenAITranscriptionSession", FakeSession), \
                patch.object(record_STT, "ToxicBertScorer", return_value=scorer), \
                patch.object(record_STT, "stream_microphone", fake_stream), \
                contextlib.redirect_stdout(output):
            text = await record_STT.record(self.args(**overrides))
        return text, output.getvalue()

    async def test_record_flags_final_turns_and_prints_summary(self):
        scorer = FakeScorer(scores={"you are useless": 0.91})
        text, output = await self.run_record(scorer)
        self.assertTrue(scorer.loaded)
        self.assertEqual(text, "you are useless\nnice round")
        self.assertIn("[HOSTILE 0.91] you are useless", output)
        self.assertEqual(output.count("[HOSTILE"), 1)
        self.assertNotIn("[CHECK", output)
        self.assertIn("Hostility check: 2 scored, 1 hostile, 0 to check, 0 errors", output)

    async def test_no_classify_skips_the_model(self):
        scorer = FakeScorer()
        text, output = await self.run_record(scorer, no_classify=True)
        self.assertFalse(scorer.loaded)
        self.assertEqual(scorer.calls, [])
        self.assertEqual(text, "you are useless\nnice round")
        self.assertNotIn("Hostility check", output)


if __name__ == "__main__":
    unittest.main()
