"""Local alignment helpers for IPA strings."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

from abydos.distance import ALINE
from lingpy import log

if hasattr(log, "setLevel"):
    log.setLevel(logging.WARNING)
logging.getLogger("lingpy").setLevel(logging.WARNING)
from lingpy.align import pw_align

AlignmentMatch = Tuple[float, str]

__all__ = [
    "AlignmentMatch",
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

    def align(self, sentence_ipa: str, query_ipa: str) -> List[AlignmentMatch]:
        if self.method == "lingpy":
            return self._align_with_lingpy(sentence_ipa, query_ipa)
        return self._align_with_aline(sentence_ipa, query_ipa)

    def _align_with_lingpy(self, sentence_ipa: str, query_ipa: str) -> List[AlignmentMatch]:
        alignment = pw_align(sentence_ipa, query_ipa, mode="overlap")
        if not alignment:
            return []
        sentence_tokens, _query_tokens, score = alignment
        matches = self._extract_sentence_matches(sentence_tokens)
        return [(score, match) for match in matches]

    def _align_with_aline(self, sentence_ipa: str, query_ipa: str) -> List[AlignmentMatch]:
        if self.aline is None:
            raise RuntimeError("ALINE aligner not initialized")
        score, sentence_alignment, _query_alignment = self.aline.alignment(
            sentence_ipa, query_ipa
        )
        matches = self._extract_sentence_matches(sentence_alignment)
        return [(score, match) for match in matches]

    @staticmethod
    def _extract_sentence_matches(
        alignment: Union[str, Sequence[str]]
    ) -> List[str]:
        tokens = LocalAlignment._normalize_alignment_tokens(alignment)
        if not tokens:
            return []

        if "‖" in tokens:
            return LocalAlignment._extract_matches_with_markers(tokens)

        cleaned = "".join(LocalAlignment._sanitize_token(token) for token in tokens)
        cleaned = cleaned.replace("‖", "")
        return [cleaned] if cleaned else []

    @staticmethod
    def _normalize_alignment_tokens(
        alignment: Union[str, Sequence[str]]
    ) -> List[str]:
        if isinstance(alignment, str):
            return [char for char in alignment if not char.isspace()]
        return list(alignment)

    @staticmethod
    def _extract_matches_with_markers(tokens: Sequence[str]) -> List[str]:
        matches: List[str] = []
        collecting = False
        current: List[str] = []
        for token in tokens:
            if token == "‖":
                if collecting:
                    cleaned = "".join(current)
                    cleaned = cleaned.replace("-", "")
                    if cleaned:
                        matches.append(cleaned)
                    current = []
                    collecting = False
                else:
                    collecting = True
                continue
            if collecting:
                cleaned_token = LocalAlignment._sanitize_token(token)
                if cleaned_token:
                    current.append(cleaned_token)
        return matches

    @staticmethod
    def _sanitize_token(token: str) -> str:
        token = token.strip()
        if not token or token == "-" or token == "‖":
            return ""
        return token


def local_align_sentence(
    sentence_ipa: str,
    query_ipas: Sequence[str],
    aligner: Optional[LocalAlignment] = None,
) -> List[Tuple[str, List[AlignmentMatch]]]:
    """Align ``sentence_ipa`` against each query in ``query_ipas``."""

    local_aligner = aligner or LocalAlignment()
    results: List[Tuple[str, List[AlignmentMatch]]] = []
    for query in query_ipas:
        results.append((query, local_aligner.align(sentence_ipa, query)))
    return results
