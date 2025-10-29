"""Local alignment helpers for IPA strings."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from abydos.distance import ALINE
from lingpy.align import pw_align

logging.getLogger("lingpy").setLevel(logging.WARNING)

# logging.getLogger("lingpy").setLevel(logging.WARNING)

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


class LocalAlignment:
    """Local alignment between a sentence IPA string and a query IPA string."""

    def __init__(self, method: str = "lingpy", aline: Optional[ALINE] = None) -> None:
        if method not in {"lingpy", "aline"}:
            raise ValueError("method must be 'lingpy' or 'aline'")
        self.method = method
        self.aline = aline or (ALINE(mode="semi-global") if method == "aline" else None)

    def align(self, sentence_ipa: str, query_ipa: str) -> AlignmentResult:
        if self.method == "lingpy":
            return self._align_with_lingpy(sentence_ipa, query_ipa)
        return self._align_with_aline(sentence_ipa, query_ipa)

    @staticmethod
    def _clean_alignment_text(text: str) -> str:
        for marker in ("-", "‖", " "):
            text = text.replace(marker, "")
        return text

    @staticmethod
    def _build_alignment_result(
        score: float, sentence_tokens: Sequence[str], query_tokens: Sequence[str]
    ) -> AlignmentResult:
        sentence_alignment = "".join(sentence_tokens)
        query_alignment = "".join(query_tokens)
        sentence_subsequence = LocalAlignment._clean_alignment_text(sentence_alignment)
        return AlignmentResult(
            score=score,
            sentence_subsequence=sentence_subsequence,
            sentence_alignment=sentence_alignment,
            query_alignment=query_alignment,
        )

    def _align_with_lingpy(self, sentence_ipa: str, query_ipa: str) -> AlignmentResult:
        alignment = pw_align(sentence_ipa, query_ipa, mode="overlap")
        if not alignment:
            return AlignmentResult(0.0, "", "", "")
        sentence_tokens, query_tokens, score = alignment
        return self._build_alignment_result(score, sentence_tokens, query_tokens)

    def _align_with_aline(self, sentence_ipa: str, query_ipa: str) -> AlignmentResult:
        if self.aline is None:
            raise RuntimeError("ALINE aligner not initialized")
        score, sentence_alignment, query_alignment = self.aline.alignment(
            sentence_ipa, query_ipa
        )
        subsequence = self._clean_alignment_text(sentence_alignment)
        return AlignmentResult(
            score=score,
            sentence_subsequence=subsequence,
            sentence_alignment=sentence_alignment,
            query_alignment=query_alignment,
        )


def local_align_sentence(
    sentence_ipa: str,
    query_ipas: Sequence[str],
    aligner: Optional[LocalAlignment] = None,
) -> List[Tuple[str, AlignmentResult]]:
    """Align ``sentence_ipa`` against each query in ``query_ipas``."""

    local_aligner = aligner or LocalAlignment()
    results: List[Tuple[str, AlignmentResult]] = []
    for query in query_ipas:
        results.append((query, local_aligner.align(sentence_ipa, query)))
    return results
