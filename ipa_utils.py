"""Utilities for converting text to IPA and performing local alignment."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from abydos.distance import ALINE
from dragonmapper.hanzi import to_ipa as hanzi_to_ipa
from eng_to_ipa import convert as eng_to_ipa_convert
from lingpy.align import we_align


_CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z]+|[^A-Za-z\u3400-\u4dbf\u4e00-\u9fff]+")


class IPAConverter:
    """Convert English and Chinese text to their IPA representation."""

    def __init__(self) -> None:
        self._token_re = _TOKEN_RE
        self._chinese_re = _CHINESE_RE

    def convert(self, text: str) -> str:
        """Convert a text containing English and Chinese characters to IPA."""
        if not text:
            return ""

        converted_segments: List[str] = []
        for segment in self._token_re.findall(text):
            if self._chinese_re.fullmatch(segment):
                converted_segments.append(self._convert_chinese(segment))
            elif segment.isalpha():
                converted_segments.append(self._convert_english(segment))
            else:
                converted_segments.append(segment)
        return "".join(converted_segments)

    @staticmethod
    def _convert_chinese(text: str) -> str:
        return hanzi_to_ipa(text)

    @staticmethod
    def _convert_english(token: str) -> str:
        if token.isupper() and len(token) > 1:
            letters = [eng_to_ipa_convert(letter).strip() for letter in token]
            letters = [letter for letter in letters if letter]
            return " ".join(letters)
        return eng_to_ipa_convert(token)


class IPALexicon:
    """Build and persist a lexicon of phrases and their IPA forms."""

    def __init__(self, converter: Optional[IPAConverter] = None) -> None:
        self.converter = converter or IPAConverter()
        self.entries: Dict[str, str] = {}

    def add_phrases(self, phrases: Iterable[str]) -> None:
        for phrase in phrases:
            if phrase is None:
                continue
            ipa_form = self.converter.convert(phrase)
            self.entries[phrase] = ipa_form

    def save_to(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.entries, file, ensure_ascii=False, indent=2)

    def load_from(self, path: Path | str) -> None:
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        self.entries = {str(key): str(value) for key, value in data.items()}


@dataclass
class AlignmentResult:
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
        self.aline = aline or (ALINE(mode="local") if method == "aline" else None)

    def align(self, sentence_ipa: str, query_ipa: str) -> AlignmentResult:
        if self.method == "lingpy":
            return self._align_with_lingpy(sentence_ipa, query_ipa)
        return self._align_with_aline(sentence_ipa, query_ipa)

    @staticmethod
    def _build_alignment_result(
        score: float, sentence_tokens: Sequence[str], query_tokens: Sequence[str]
    ) -> AlignmentResult:
        sentence_alignment = "".join(sentence_tokens)
        query_alignment = "".join(query_tokens)
        sentence_subsequence = sentence_alignment.replace("-", "")
        return AlignmentResult(
            score=score,
            sentence_subsequence=sentence_subsequence,
            sentence_alignment=sentence_alignment,
            query_alignment=query_alignment,
        )

    def _align_with_lingpy(self, sentence_ipa: str, query_ipa: str) -> AlignmentResult:
        alignments = we_align(sentence_ipa, query_ipa)
        if not alignments:
            return AlignmentResult(0.0, "", "", "")
        sentence_tokens, query_tokens, score = alignments[0]
        return self._build_alignment_result(score, sentence_tokens, query_tokens)

    def _align_with_aline(self, sentence_ipa: str, query_ipa: str) -> AlignmentResult:
        if self.aline is None:
            raise RuntimeError("ALINE aligner not initialized")
        score, sentence_alignment, query_alignment = self.aline.alignment(
            sentence_ipa, query_ipa
        )
        subsequence = self._extract_subsequence(sentence_alignment)
        return AlignmentResult(
            score=score,
            sentence_subsequence=subsequence,
            sentence_alignment=sentence_alignment,
            query_alignment=query_alignment,
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
    sentence_ipa: str,
    query_ipas: Sequence[str],
    aligner: Optional[LocalAlignment] = None,
) -> List[Tuple[str, AlignmentResult]]:
    local_aligner = aligner or LocalAlignment()
    results: List[Tuple[str, AlignmentResult]] = []
    for query in query_ipas:
        result = local_aligner.align(sentence_ipa, query)
        results.append((query, result))
    return results
