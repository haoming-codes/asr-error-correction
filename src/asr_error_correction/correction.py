"""ASR correction utilities driven by phonetic alignment."""
from __future__ import annotations

import itertools
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Iterator, List, Sequence, Tuple

from .alignment import LocalAlignment
from .lexicon import IPALexicon
from .models import AlignmentResult, GraphemePhoneme, ReplacementPlan

__all__ = ["ASRCorrector"]


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

        candidates = self._candidate_replacements(sentence_gp)
        if not candidates:
            return sentence_gp.grapheme_str

        best_score = max(candidate.score for candidate in candidates)
        best_matches = (
            candidate for candidate in candidates if candidate.score == best_score
        )

        filtered_matches = self._filter_overlaps(best_matches)
        if not filtered_matches:
            return sentence_gp.grapheme_str

        return self._apply_replacements(sentence_gp.grapheme_str, filtered_matches)

    def candidate_replacements(self, sentence: str) -> List[ReplacementPlan]:
        """Return all candidate replacements for ``sentence`` above the threshold."""

        if not sentence:
            return []

        sentence_gp = self.lexicon.converter.convert_to_grapheme_phoneme(sentence)
        return self._candidate_replacements(sentence_gp)

    def _candidate_replacements(
        self, sentence_gp: GraphemePhoneme
    ) -> List[ReplacementPlan]:
        entries = list(self.lexicon.entries.values())
        if not entries:
            return []

        candidates = list(self._iter_replacements(sentence_gp, entries))
        return sorted(
            candidates, key=lambda match: (-match.score, match.start, match.end)
        )

    def _iter_replacements(
        self,
        sentence_gp: GraphemePhoneme,
        entries: Sequence[GraphemePhoneme],
    ) -> Iterator[ReplacementPlan]:
        if self._should_parallelize(sentence_gp, entries):
            with ProcessPoolExecutor(max_workers=self.use_concurrency) as executor:
                for replacements in executor.map(
                    ASRCorrector._align_entry,
                    itertools.repeat(sentence_gp),
                    entries,
                    itertools.repeat(self.score_threshold),
                    itertools.repeat(self.aligner),
                ):
                    yield from replacements
        else:
            for entry in entries:
                yield from self._align_entry(
                    sentence_gp, entry, self.score_threshold, self.aligner
                )

    def _should_parallelize(
        self, sentence_gp: GraphemePhoneme, entries: Sequence[GraphemePhoneme]
    ) -> bool:
        if self.use_concurrency <= 1 or len(entries) <= 1:
            return False
        try:
            pickle.dumps((sentence_gp, self.aligner))
        except (pickle.PicklingError, AttributeError, TypeError):
            return False
        return True

    @staticmethod
    def _align_entry(
        sentence_gp: GraphemePhoneme,
        entry: GraphemePhoneme,
        score_threshold: float,
        aligner: LocalAlignment,
    ) -> List[ReplacementPlan]:
        replacements: List[ReplacementPlan] = []
        for raw_result in aligner.align(sentence_gp, entry):
            result = ASRCorrector._coerce_alignment_result(raw_result)
            if result.score < score_threshold:
                continue
            replacements.append(
                ReplacementPlan(
                    score=result.score,
                    start=result.start,
                    end=result.end,
                    replacement=entry,
                )
            )
        return replacements

    @staticmethod
    def _filter_overlaps(
        matches: Iterable[ReplacementPlan],
    ) -> List[ReplacementPlan]:
        ordered = sorted(matches, key=lambda match: (match.start, -match.end))
        filtered: List[ReplacementPlan] = []
        current_end = -1
        for match in ordered:
            if match.start < current_end:
                continue
            filtered.append(match)
            current_end = match.end
        return filtered

    @staticmethod
    def _apply_replacements(
        text: str, matches: Sequence[ReplacementPlan]
    ) -> str:
        cursor = 0
        parts: List[str] = []
        for match in matches:
            parts.append(text[cursor : match.start])
            parts.append(match.replacement.grapheme_str)
            cursor = match.end
        parts.append(text[cursor:])
        return "".join(parts)

    @staticmethod
    def _coerce_alignment_result(
        result: AlignmentResult | Tuple[float, int, int, GraphemePhoneme]
    ) -> AlignmentResult:
        if isinstance(result, AlignmentResult):
            return result
        score, start, end, match = result
        if not isinstance(match, GraphemePhoneme):  # pragma: no cover - safety net
            raise TypeError("Alignment result must contain a GraphemePhoneme match")
        return AlignmentResult(score=score, start=start, end=end, match=match)

