"""ASR correction utilities driven by phonetic alignment."""
from __future__ import annotations

import itertools
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .alignment import LocalAlignment
from .conversion import GraphemePhoneme
from .lexicon import IPALexicon

__all__ = ["ASRCorrector"]


@dataclass(frozen=True)
class _Replacement:
    """Description of a replacement within a sentence string."""

    score: float
    start: int
    end: int
    replacement: GraphemePhoneme


class ASRCorrector:
    """Apply phonetic alignment against a lexicon to correct ASR output."""

    def __init__(
        self,
        lexicon: IPALexicon,
        *,
        score_threshold: float = 0.0,
        aligner: LocalAlignment | None = None,
        use_concurrency: int = 32,
    ) -> None:
        self.lexicon = lexicon
        self.score_threshold = score_threshold
        self.aligner = aligner or LocalAlignment()
        if use_concurrency < 0:
            raise ValueError("use_concurrency must be non-negative")
        self.use_concurrency = use_concurrency

    def correct(self, sentence: str) -> str:
        """Return ``sentence`` with high-scoring alignments replaced."""

        if not sentence:
            return sentence

        sentence_gp = self.lexicon.converter.convert_to_grapheme_phoneme(sentence)

        best_matches = self._collect_best_matches(sentence_gp)
        if not best_matches:
            return sentence_gp.grapheme_str

        return self._apply_replacements(sentence_gp.grapheme_str, best_matches)

    def _collect_best_matches(
        self, sentence_gp: GraphemePhoneme
    ) -> Sequence[_Replacement]:
        entries = list(self.lexicon.entries.values())
        if not entries:
            return []

        candidates: List[_Replacement] = []
        can_parallelize = self.use_concurrency > 1 and len(entries) > 1
        if can_parallelize:
            try:
                pickle.dumps((sentence_gp, self.aligner))
            except (pickle.PicklingError, AttributeError, TypeError):
                can_parallelize = False

        if can_parallelize:
            max_workers = self.use_concurrency
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for replacements in executor.map(
                    ASRCorrector._align_entry,
                    itertools.repeat(sentence_gp),
                    entries,
                    itertools.repeat(self.score_threshold),
                    itertools.repeat(self.aligner),
                ):
                    candidates.extend(replacements)
        else:
            for entry in entries:
                candidates.extend(
                    self._align_entry(
                        sentence_gp, entry, self.score_threshold, self.aligner
                    )
                )

        if not candidates:
            return []

        best_score = max(candidate.score for candidate in candidates)
        best_matches = [
            candidate for candidate in candidates if candidate.score == best_score
        ]

        return self._filter_overlaps(best_matches)

    @staticmethod
    def _align_entry(
        sentence_gp: GraphemePhoneme,
        entry: GraphemePhoneme,
        score_threshold: float,
        aligner: LocalAlignment,
    ) -> List[_Replacement]:
        replacements: List[_Replacement] = []
        for score, start, end, _ in aligner.align(sentence_gp, entry):
            if score < score_threshold:
                continue
            replacements.append(_Replacement(score, start, end, entry))
        return replacements

    @staticmethod
    def _filter_overlaps(matches: Iterable[_Replacement]) -> List[_Replacement]:
        ordered = sorted(matches, key=lambda match: (match.start, -match.end))
        filtered: List[_Replacement] = []
        current_end = -1
        for match in ordered:
            if match.start < current_end:
                continue
            filtered.append(match)
            current_end = match.end
        return filtered

    @staticmethod
    def _apply_replacements(text: str, matches: Sequence[_Replacement]) -> str:
        cursor = 0
        parts: List[str] = []
        for match in matches:
            parts.append(text[cursor : match.start])
            parts.append(match.replacement.grapheme_str)
            cursor = match.end
        parts.append(text[cursor:])
        return "".join(parts)

