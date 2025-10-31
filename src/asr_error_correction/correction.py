"""ASR correction utilities driven by phonetic alignment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ) -> None:
        self.lexicon = lexicon
        self.score_threshold = score_threshold
        self.aligner = aligner or LocalAlignment()

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

        best_score: float | None = None
        best_matches: List[_Replacement] = []

        max_workers = min(32, len(entries)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.aligner.align, sentence_gp, entry): entry
                for entry in entries
            }
            for future in as_completed(futures):
                entry = futures[future]
                for score, start, end, _ in future.result():
                    if score < self.score_threshold:
                        continue
                    if best_score is None or score > best_score:
                        best_score = score
                        best_matches = []
                    if best_score is not None and score == best_score:
                        best_matches.append(_Replacement(score, start, end, entry))

        return self._filter_overlaps(best_matches)

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

