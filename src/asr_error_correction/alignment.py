"""Local alignment helpers for IPA strings."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from abydos.distance import ALINE
from lingpy.align import we_align

from .conversion import GraphemePhoneme
logging.getLogger("lingpy").setLevel(logging.WARNING)

__all__ = [
    "AlignmentResult",
    "LocalAlignment",
    "local_align_sentence",
]


@dataclass
class AlignmentResult:
    """Result container for a local alignment operation."""

    score: float
    sentence_subsequence: str
    sentence_alignment: str
    query_alignment: str
    sentence: GraphemePhoneme
    query: GraphemePhoneme

    @property
    def sentence_graphemes(self) -> str:
        """Return the grapheme form of the aligned sentence."""

        return self.sentence.graphemes

    @property
    def query_graphemes(self) -> str:
        """Return the grapheme form of the aligned query."""

        return self.query.graphemes


class LocalAlignment:
    """Local alignment between a sentence IPA string and a query IPA string."""

    def __init__(self, method: str = "lingpy", aline: Optional[ALINE] = None) -> None:
        if method not in {"lingpy", "aline"}:
            raise ValueError("method must be 'lingpy' or 'aline'")
        self.method = method
        self.aline = aline or (ALINE(mode="local") if method == "aline" else None)

    def align(
        self, sentence: GraphemePhoneme, query: GraphemePhoneme
    ) -> AlignmentResult:
        if self.method == "lingpy":
            return self._align_with_lingpy(sentence, query)
        return self._align_with_aline(sentence, query)

    @staticmethod
    def _build_alignment_result(
        score: float,
        sentence_tokens: Sequence[str],
        query_tokens: Sequence[str],
        sentence: GraphemePhoneme,
        query: GraphemePhoneme,
    ) -> AlignmentResult:
        sentence_alignment = "".join(sentence_tokens)
        query_alignment = "".join(query_tokens)
        sentence_subsequence = sentence_alignment.replace("-", "")
        return AlignmentResult(
            score=score,
            sentence_subsequence=sentence_subsequence,
            sentence_alignment=sentence_alignment,
            query_alignment=query_alignment,
            sentence=sentence,
            query=query,
        )

    def _align_with_lingpy(
        self, sentence: GraphemePhoneme, query: GraphemePhoneme
    ) -> AlignmentResult:
        alignments = we_align(sentence.phonemes, query.phonemes)
        if not alignments:
            return AlignmentResult(0.0, "", "", "", sentence, query)
        sentence_tokens, query_tokens, score = alignments[0]
        return self._build_alignment_result(
            score, sentence_tokens, query_tokens, sentence, query
        )

    def _align_with_aline(
        self, sentence: GraphemePhoneme, query: GraphemePhoneme
    ) -> AlignmentResult:
        if self.aline is None:
            raise RuntimeError("ALINE aligner not initialized")
        score, sentence_alignment, query_alignment = self.aline.alignment(
            sentence.phonemes, query.phonemes
        )
        subsequence = self._extract_subsequence(sentence_alignment)
        return AlignmentResult(
            score=score,
            sentence_subsequence=subsequence,
            sentence_alignment=sentence_alignment,
            query_alignment=query_alignment,
            sentence=sentence,
            query=query,
        )

    @staticmethod
    def _extract_subsequence(alignment: str) -> str:
        if "‖" in alignment:
            parts = alignment.split("‖")
            if len(parts) >= 3:
                segment = parts[1]
            else:
                segment = alignment
        else:
            segment = alignment
        return "".join(segment.split())


def local_align_sentence(
    sentence: GraphemePhoneme,
    query_items: Sequence[GraphemePhoneme],
    aligner: Optional[LocalAlignment] = None,
) -> List[Tuple[GraphemePhoneme, AlignmentResult]]:
    """Align ``sentence`` against each query in ``query_items``."""

    local_aligner = aligner or LocalAlignment()
    results: List[Tuple[GraphemePhoneme, AlignmentResult]] = []
    for query in query_items:
        results.append((query, local_aligner.align(sentence, query)))
    return results
