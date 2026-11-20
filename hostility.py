"""Score final spoken turns for possible hostility without blocking transcription.

The default scorer is a placeholder: toxic-bert measures general toxicity, not this
project's hostile-intent labels, and the default thresholds are provisional.
"""

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
import re
import sys
import time
from typing import Callable, Protocol

LOCAL_SPEAKER = "local"
CONTEXT_TURNS = 5
CHECK_THRESHOLD = 0.3
FLAG_THRESHOLD = 0.7


@dataclass(frozen=True)
class ContextTurn:
    speaker_id: str
    text: str


@dataclass(frozen=True)
class Decision:
    item_id: str
    speaker_id: str
    text: str
    score: float
    level: str
    latency_ms: float


class Scorer(Protocol):
    def load(self) -> None: ...

    def score(self, text: str, context: list[ContextTurn]) -> float: ...


class ToxicBertScorer:
    """Placeholder scorer; it reads the current text only and ignores context."""

    model_name = "unitary/toxic-bert"

    def __init__(self):
        self._classifier = None

    def load(self) -> None:
        # Imported here so transcription-only runs skip the slow model libraries.
        import torch
        from transformers import pipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._classifier = pipeline(
            "text-classification", model=self.model_name, top_k=None, device=device,
        )

    def score(self, text: str, context: list[ContextTurn]) -> float:
        if self._classifier is None:
            raise RuntimeError("Load the scorer before scoring text.")
        results = self._classifier(text, truncation=True, function_to_apply="sigmoid")
        labels = results[0] if isinstance(results[0], list) else results
        return next(float(item["score"]) for item in labels if item["label"] == "toxic")


def level_for(score: float, check: float = CHECK_THRESHOLD, flag: float = FLAG_THRESHOLD) -> str:
    if score >= flag:
        return "hostile"
    return "check" if score >= check else "ignore"


def split_turn(text: str, max_words: int = 25) -> list[str]:
    """Keep short turns whole; split long turns on sentence punctuation."""
    text = text.strip()
    if not text:
        return []
    if len(text.split()) <= max_words:
        return [text]
    return [part for part in re.split(r"(?<=[.!?])\s+", text) if part]


def display_decision(decision: Decision) -> None:
    if decision.level != "ignore":
        print(f"[{decision.level.upper()} {decision.score:.2f}] {decision.text}", flush=True)


class HostilityMonitor:
    """Queue final turns in spoken order and score them on a worker thread."""

    def __init__(
        self,
        scorer: Scorer,
        *,
        check: float = CHECK_THRESHOLD,
        flag: float = FLAG_THRESHOLD,
        on_decision: Callable[[Decision], None] = display_decision,
        speaker_id: str = LOCAL_SPEAKER,
    ):
        self._scorer = scorer
        self._check = check
        self._flag = flag
        self._on_decision = on_decision
        self.speaker_id = speaker_id
        self.decisions: list[Decision] = []
        self.errors = 0
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._context: deque[ContextTurn] = deque(maxlen=CONTEXT_TURNS)
        self._cursor = 0

    def release(self, session, *, skip_unfinished: bool = False) -> None:
        """Queue finished turns, waiting for earlier turns unless told to skip them."""
        while self._cursor < len(session.item_order):
            item_id = session.item_order[self._cursor]
            transcript = session.transcripts.get(item_id)
            if transcript is None or not transcript.final:
                if not skip_unfinished:
                    return
                self._cursor += 1
                continue
            self._cursor += 1
            for part in split_turn(transcript.text):
                self._queue.put_nowait((item_id, part))

    async def run(self) -> None:
        """Score queued turns until cancelled; one bad turn never stops transcription."""
        while True:
            item_id, text = await self._queue.get()
            try:
                started = time.perf_counter()
                score = await asyncio.to_thread(self._scorer.score, text, list(self._context))
                decision = Decision(
                    item_id, self.speaker_id, text, score,
                    level_for(score, self._check, self._flag),
                    (time.perf_counter() - started) * 1000,
                )
                self.decisions.append(decision)
                self._on_decision(decision)
            except Exception as exc:
                self.errors += 1
                print(f"Hostility scoring skipped a turn: {exc}", file=sys.stderr, flush=True)
            finally:
                self._context.append(ContextTurn(self.speaker_id, text))
                self._queue.task_done()

    async def finish(self, session, timeout: float = 30) -> None:
        """Score every remaining final turn before the session closes."""
        self.release(session, skip_unfinished=True)
        try:
            await asyncio.wait_for(self._queue.join(), timeout)
        except asyncio.TimeoutError:
            print("Hostility scoring timed out; some turns were not scored.",
                  file=sys.stderr, flush=True)

    def summary(self) -> str:
        levels = Counter(decision.level for decision in self.decisions)
        latencies = [decision.latency_ms for decision in self.decisions]
        timing = (f"; latency mean {sum(latencies) / len(latencies):.0f} ms, "
                  f"max {max(latencies):.0f} ms") if latencies else ""
        return (f"Hostility check: {len(self.decisions)} scored, {levels['hostile']} hostile, "
                f"{levels['check']} to check, {self.errors} errors{timing}")
