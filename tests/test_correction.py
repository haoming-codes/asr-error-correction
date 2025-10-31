"""Tests for ASR correction parallel alignment collection."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Tuple

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from asr_error_correction.correction import ASRCorrector
from asr_error_correction.conversion import GraphemePhoneme
from asr_error_correction.lexicon import IPALexicon


class _BarrierAligner:
    """Alignment helper that blocks until all threads reach a barrier."""

    def __init__(
        self,
        barrier: threading.Barrier,
        responses: Dict[str, List[Tuple[float, int, int, GraphemePhoneme]]],
    ) -> None:
        self._barrier = barrier
        self._responses = responses

    def align(
        self, sentence: GraphemePhoneme, query: GraphemePhoneme
    ) -> List[Tuple[float, int, int, GraphemePhoneme]]:
        try:
            self._barrier.wait(timeout=5)
        except threading.BrokenBarrierError as exc:  # pragma: no cover - defensive
            raise AssertionError("Alignment did not execute concurrently") from exc
        return self._responses[query.grapheme_str]


def test_collect_best_matches_uses_parallel_alignment() -> None:
    sentence = GraphemePhoneme.from_components(
        grapheme_str="abcd",
        grapheme_list=["ab", "c", "d"],
        phoneme_str="abcd",
        phoneme_list=["ab", "c", "d"],
    )

    entry_a = GraphemePhoneme.from_components(
        grapheme_str="alpha",
        grapheme_list=["alpha"],
        phoneme_str="alpha",
        phoneme_list=["alpha"],
    )
    entry_b = GraphemePhoneme.from_components(
        grapheme_str="beta",
        grapheme_list=["beta"],
        phoneme_str="beta",
        phoneme_list=["beta"],
    )
    entry_c = GraphemePhoneme.from_components(
        grapheme_str="gamma",
        grapheme_list=["gamma"],
        phoneme_str="gamma",
        phoneme_list=["gamma"],
    )

    lexicon = IPALexicon()
    lexicon.entries = {
        "alpha": entry_a,
        "beta": entry_b,
        "gamma": entry_c,
    }

    barrier = threading.Barrier(parties=len(lexicon.entries))
    responses = {
        "alpha": [(1.0, 0, 1, sentence)],
        "beta": [(2.0, 1, 2, sentence)],
        "gamma": [(2.0, 2, 4, sentence)],
    }
    aligner = _BarrierAligner(barrier, responses)

    corrector = ASRCorrector(lexicon, aligner=aligner)

    matches = corrector._collect_best_matches(sentence)

    assert {match.replacement.grapheme_str for match in matches} == {"beta", "gamma"}
    assert all(match.score == 2.0 for match in matches)


def test_collect_best_matches_empty_lexicon() -> None:
    sentence = GraphemePhoneme.from_components(
        grapheme_str="hello",
        grapheme_list=["hello"],
        phoneme_str="hello",
        phoneme_list=["hello"],
    )

    lexicon = IPALexicon()
    corrector = ASRCorrector(lexicon)

    assert corrector._collect_best_matches(sentence) == []
