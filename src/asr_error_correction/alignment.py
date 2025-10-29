"""Local alignment helpers for IPA strings."""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from lingpy.align import pw_align

logging.getLogger("lingpy").setLevel(logging.WARNING)

__all__ = [
    "LocalAlignment",
    "local_align_sentence",
]


def _is_alignment_marker(token: str) -> bool:
    """Return ``True`` if *token* represents an insertion/deletion marker."""

    if not token:
        return True
    stripped = token.strip()
    return stripped in {"", "-", "‖"}


def _extract_matched_subsequence(
    sentence_tokens: Sequence[str], query_tokens: Sequence[str]
) -> str:
    """Return the subsequence from ``sentence_tokens`` aligned with ``query_tokens``."""

    matched: List[str] = []
    for sentence_token, query_token in zip(sentence_tokens, query_tokens):
        if _is_alignment_marker(query_token) or _is_alignment_marker(sentence_token):
            continue
        matched.append(sentence_token)
    return "".join(matched)


class LocalAlignment:
    """Local alignment between a sentence IPA string and a query IPA string."""

    def align(self, sentence_ipa: str, query_ipa: str) -> List[Tuple[float, str]]:
        """Align ``sentence_ipa`` against ``query_ipa`` using ``lingpy``."""

        alignment = pw_align(sentence_ipa, query_ipa, mode="overlap")
        if not alignment:
            return []
        sentence_tokens, query_tokens, score = alignment
        matched = _extract_matched_subsequence(sentence_tokens, query_tokens)
        if not matched:
            return []
        return [(score, matched)]


def local_align_sentence(
    sentence_ipa: str,
    query_ipas: Sequence[str],
    aligner: LocalAlignment | None = None,
) -> List[Tuple[str, List[Tuple[float, str]]]]:
    """Align ``sentence_ipa`` against each query in ``query_ipas``."""

    local_aligner = aligner or LocalAlignment()
    results: List[Tuple[str, List[Tuple[float, str]]]] = []
    for query in query_ipas:
        results.append((query, local_aligner.align(sentence_ipa, query)))
    return results
