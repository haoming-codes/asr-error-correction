"""Tools for converting mixed English/Chinese text into IPA."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from dragonmapper.hanzi import to_ipa as hanzi_to_ipa
from eng_to_ipa import convert as eng_to_ipa_convert


__all__ = ["GraphemePhoneme", "IPAConverter", "TokenizedSegment"]


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


_STRESS_TRANSLATION = str.maketrans("", "", "ˈˌ")
_TONE_TRANSLATION = str.maketrans("", "", "012345˥˦˧˨˩˩˨˧˦˥")
_WHITESPACE_RE = re.compile(r"\s+")


class IPAConverter:
    """Convert English and Chinese text to their IPA representations."""

    def __init__(
        self,
        *,
        remove_tone_marks: bool = False,
        remove_stress_marks: bool = False,
        strip_whitespace: bool = False,
        remove_punctuation: bool = False,
    ) -> None:
        self.remove_tone_marks = remove_tone_marks
        self.remove_stress_marks = remove_stress_marks
        self.strip_whitespace = strip_whitespace
        self.remove_punctuation = remove_punctuation

    def tokenize(self, text: str) -> Iterable[TokenizedSegment]:
        """Yield the detected segments from ``text``."""
        for match in _TOKEN_RE.findall(text):
            yield TokenizedSegment(match)

    def convert(self, text: str) -> str:
        """Convert a text containing English and Chinese characters to IPA."""
        if not text:
            return ""

        converted: List[Tuple[str, bool]] = []
        for token in self.tokenize(text):
            value = self._convert_segment(token)
            is_ipa_segment = token.is_alpha or token.is_chinese
            if self.remove_punctuation and not is_ipa_segment:
                value = self._remove_punctuation(value)
                if not value:
                    continue
            converted.append((value, is_ipa_segment))

        processed: List[str] = []
        for value, is_ipa_segment in converted:
            if is_ipa_segment:
                value = self._apply_marker_options(value)
            processed.append(value)

        result = "".join(processed)
        if self.strip_whitespace:
            result = _WHITESPACE_RE.sub("", result)
        return result

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

    def _apply_marker_options(self, value: str) -> str:
        if self.remove_stress_marks:
            value = value.translate(_STRESS_TRANSLATION)
        if self.remove_tone_marks:
            value = value.translate(_TONE_TRANSLATION)
        return value

    @staticmethod
    def _remove_punctuation(value: str) -> str:
        return "".join(
            ch for ch in value if not unicodedata.category(ch).startswith("P")
        )


@dataclass(frozen=True)
class GraphemePhoneme:
    """Representation of a sentence in both grapheme and phoneme form."""

    graphemes: str
    _phonemes: str

    def __init__(
        self,
        graphemes: str,
        *,
        converter: Optional[IPAConverter] = None,
        phonemes: Optional[str] = None,
    ) -> None:
        object.__setattr__(self, "graphemes", graphemes)
        if phonemes is None:
            ipa_converter = converter or IPAConverter()
            phonemes = ipa_converter.convert(graphemes)
        object.__setattr__(self, "_phonemes", phonemes)

    @property
    def phonemes(self) -> str:
        """Return the IPA phoneme representation for this sentence."""

        return self._phonemes

    def to_mapping(self) -> Tuple[str, str]:
        """Return a tuple of grapheme and phoneme representations."""

        return self.graphemes, self._phonemes

    @classmethod
    def from_mapping(
        cls,
        graphemes: str,
        phonemes: str,
        *,
        converter: Optional[IPAConverter] = None,
    ) -> GraphemePhoneme:
        """Rebuild an instance from previously persisted data."""

        return cls(graphemes, converter=converter, phonemes=phonemes)
