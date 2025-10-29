"""Tools for converting mixed English/Chinese text into IPA."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from dragonmapper.hanzi import to_ipa as hanzi_to_ipa
from eng_to_ipa import convert as eng_to_ipa_convert


__all__ = ["IPAConverter", "TokenizedSegment"]


_CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z]+|[^A-Za-z\u3400-\u4dbf\u4e00-\u9fff]+"
)


@dataclass(frozen=True)
class TokenizedSegment:
    """A token extracted from the input string prior to conversion."""

    raw: str

    @property
    def is_chinese(self) -> bool:
        return _CHINESE_RE.fullmatch(self.raw) is not None

    @property
    def is_alpha(self) -> bool:
        return self.raw.isalpha()


class IPAConverter:
    """Convert English and Chinese text to their IPA representations."""

    def tokenize(self, text: str) -> Iterable[TokenizedSegment]:
        """Yield the detected segments from ``text``."""
        for match in _TOKEN_RE.findall(text):
            yield TokenizedSegment(match)

    def convert(self, text: str) -> str:
        """Convert a text containing English and Chinese characters to IPA."""
        if not text:
            return ""

        converted: List[str] = []
        for token in self.tokenize(text):
            converted.append(self._convert_segment(token))
        return "".join(converted)

    @staticmethod
    def _convert_segment(segment: TokenizedSegment) -> str:
        if segment.is_chinese:
            return IPAConverter._convert_chinese(segment.raw)
        if segment.is_alpha:
            return IPAConverter._convert_english(segment.raw)
        return segment.raw

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
