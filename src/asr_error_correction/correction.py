"""ASR correction utilities driven by phonetic alignment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
        use_concurrency: bool = True,
    ) -> None:
        self.lexicon = lexicon
        self.score_threshold = score_threshold
        self.aligner = aligner or LocalAlignment()
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

        def _align_entry(entry: GraphemePhoneme) -> List[_Replacement]:
            replacements: List[_Replacement] = []
            for score, start, end, _ in self.aligner.align(sentence_gp, entry):
                if score < self.score_threshold:
                    continue
                replacements.append(_Replacement(score, start, end, entry))
            return replacements

        candidates: List[_Replacement] = []
        if self.use_concurrency and len(entries) > 1:
            max_workers = min(32, len(entries)) or None
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for replacements in executor.map(_align_entry, entries):
                    candidates.extend(replacements)
        else:
            for entry in entries:
                candidates.extend(_align_entry(entry))

        if not candidates:
            return []

        best_score = max(candidate.score for candidate in candidates)
        best_matches = [
            candidate for candidate in candidates if candidate.score == best_score
        ]

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

