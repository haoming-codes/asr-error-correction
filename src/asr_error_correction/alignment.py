"""Local alignment helpers for IPA strings."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from abydos.distance import ALINE
from lingpy import log

if hasattr(log, "setLevel"):
    log.setLevel(logging.WARNING)
else:
    logging.getLogger("lingpy").setLevel(logging.WARNING)
from lingpy.align import pw_align

# logging.getLogger("lingpy").setLevel(logging.WARNING)

__all__ = [
    "LocalAlignment",
    "local_align_sentence",
]


class LocalAlignment:
    """Local alignment between a sentence IPA string and a query IPA string."""

    def __init__(self, method: str = "lingpy", aline: Optional[ALINE] = None) -> None:
        if method not in {"lingpy", "aline"}:
            raise ValueError("method must be 'lingpy' or 'aline'")
        self.method = method
        self.aline = aline or (ALINE(mode="semi-global") if method == "aline" else None)

    def align(self, sentence_ipa: str, query_ipa: str) -> List[Tuple[float, str]]:
        if self.method == "lingpy":
            return self._align_with_lingpy(sentence_ipa, query_ipa)
        return self._align_with_aline(sentence_ipa, query_ipa)

    @staticmethod
    def _clean_alignment_segment(alignment: str) -> str:
        if "‖" in alignment:
            parts = alignment.split("‖")
            if len(parts) >= 3:
                alignment = parts[1]
            else:
                alignment = alignment.replace("‖", "")
        markers = {"-", "‖", " "}
        return "".join(char for char in alignment if char not in markers)

    @classmethod
    def _clean_lingpy_tokens(cls, tokens: Sequence[str]) -> str:
        return cls._clean_alignment_segment("".join(tokens))

    def _align_with_lingpy(self, sentence_ipa: str, query_ipa: str) -> List[Tuple[float, str]]:
        alignments = pw_align(sentence_ipa, query_ipa, mode="overlap")
        results: List[Tuple[float, str]] = []
        for sentence_tokens, _query_tokens, score in alignments:
            subsequence = self._clean_lingpy_tokens(sentence_tokens)
            results.append((score, subsequence))
        return results

    def _align_with_aline(self, sentence_ipa: str, query_ipa: str) -> List[Tuple[float, str]]:
        if self.aline is None:
            raise RuntimeError("ALINE aligner not initialized")
        results: List[Tuple[float, str]] = []
        for score, sentence_alignment, _query_alignment in self.aline.alignments(
            sentence_ipa, query_ipa
        ):
            subsequence = self._clean_alignment_segment(sentence_alignment)
            results.append((score, subsequence))
        return results


def local_align_sentence(
    sentence_ipa: str,
    query_ipas: Sequence[str],
    aligner: Optional[LocalAlignment] = None,
) -> List[Tuple[str, List[Tuple[float, str]]]]:
    """Align ``sentence_ipa`` against each query in ``query_ipas``."""

    local_aligner = aligner or LocalAlignment()
    results: List[Tuple[str, List[Tuple[float, str]]]] = []
    for query in query_ipas:
        results.append((query, local_aligner.align(sentence_ipa, query)))
    return results
