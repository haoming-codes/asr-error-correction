"""Local alignment helpers for IPA strings."""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from lingpy.align import pw_align
from lingpy.sequence.sound_classes import ipa2tokens

from .conversion import GraphemePhoneme

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


def _extract_matched_span(
    sentence_tokens: Sequence[str], query_tokens: Sequence[str]
) -> Tuple[str, int, int] | None:
    """Return the subsequence and bounds from ``sentence_tokens``."""

    matched: List[str] = []
    start_index: int | None = None
    sentence_index = 0

    for sentence_token, query_token in zip(sentence_tokens, query_tokens):
        if _is_alignment_marker(sentence_token):
            continue

        if not _is_alignment_marker(query_token):
            if start_index is None:
                start_index = sentence_index
            matched.append(sentence_token)

        sentence_index += len(sentence_token)

    if not matched or start_index is None:
        return None

    matched_str = "".join(matched)
    end_index = start_index + len(matched_str)
    return matched_str, start_index, end_index


class LocalAlignment:
    """Local alignment between a sentence IPA string and a query IPA string."""

    def align(
        self, sentence: GraphemePhoneme, query: GraphemePhoneme
    ) -> List[Tuple[float, int, int, GraphemePhoneme]]:
        """Align ``sentence`` against ``query`` using ``lingpy``."""

        alignment = pw_align(ipa2tokens(sentence.phoneme_str), ipa2tokens(query.phoneme_str), mode="overlap")
        if not alignment:
            return []
        sentence_tokens, query_tokens, score = alignment
        span = _extract_matched_span(sentence_tokens, query_tokens)
        if not span:
            return []
        _, phoneme_start, phoneme_end = span

        indices = [
            idx
            for idx, (token_start, token_end) in enumerate(sentence.phoneme_spans)
            if token_end > phoneme_start and token_start < phoneme_end
        ]
        if not indices:
            return []
        first_idx, last_idx = indices[0], indices[-1]
        grapheme_start, _ = sentence.grapheme_spans[first_idx]
        _, grapheme_end = sentence.grapheme_spans[last_idx]

        matched = sentence.subsequence_covering_span(phoneme_start, phoneme_end)
        if matched is None:
            return []
        return [(score, grapheme_start, grapheme_end, matched)]


def local_align_sentence(
    sentence: GraphemePhoneme,
    query_graphemes: Sequence[GraphemePhoneme],
    aligner: LocalAlignment | None = None,
) -> List[
    Tuple[GraphemePhoneme, List[Tuple[float, int, int, GraphemePhoneme]]]
]:
    """Align ``sentence`` against each query in ``query_graphemes``."""

    local_aligner = aligner or LocalAlignment()
    results: List[
        Tuple[GraphemePhoneme, List[Tuple[float, int, int, GraphemePhoneme]]]
    ] = []
    for query in query_graphemes:
        results.append((query, local_aligner.align(sentence, query)))
    return results
