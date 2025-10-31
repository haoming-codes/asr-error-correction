"""Tools for converting mixed English/Chinese text into IPA."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Tuple

from dragonmapper.hanzi import to_ipa as hanzi_to_ipa
from eng_to_ipa import convert as eng_to_ipa_convert

from .models import GraphemePhoneme
from .tokenization import TokenizedSegment, tokenize_mixed_text


__all__ = ["GraphemePhoneme", "IPAConverter", "TokenizedSegment"]


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
        yield from tokenize_mixed_text(text)

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

    def convert_to_grapheme_phoneme(self, text: str) -> GraphemePhoneme:
        """Convert *text* into a :class:`GraphemePhoneme` description."""

        grapheme_list: List[str] = []
        phoneme_list: List[str] = []
        grapheme_spans: List[Tuple[int, int]] = []
        phoneme_spans: List[Tuple[int, int]] = []
        phoneme_cursor = 0

        for token in self.tokenize(text):
            if token.is_chinese:
                for offset, character in enumerate(token.raw):
                    phoneme = self._convert_chinese(character)
                    phoneme = self._apply_marker_options(phoneme)
                    sanitized = self._sanitize_phoneme(phoneme)
                    if not sanitized:
                        continue
                    grapheme_list.append(character)
                    start = token.start + offset
                    grapheme_spans.append((start, start + 1))
                    phoneme_list.append(sanitized)
                    phoneme_spans.append((phoneme_cursor, phoneme_cursor + len(sanitized)))
                    phoneme_cursor += len(sanitized)
            elif token.is_alpha:
                phoneme = self._convert_english(token.raw)
                phoneme = self._apply_marker_options(phoneme)
                sanitized = self._sanitize_phoneme(phoneme)
                if not sanitized:
                    continue
                grapheme_list.append(token.raw)
                grapheme_spans.append((token.start, token.end))
                phoneme_list.append(sanitized)
                phoneme_spans.append((phoneme_cursor, phoneme_cursor + len(sanitized)))
                phoneme_cursor += len(sanitized)

        return GraphemePhoneme(
            grapheme_str=text,
            grapheme_list=tuple(grapheme_list),
            phoneme_str="".join(phoneme_list),
            phoneme_list=tuple(phoneme_list),
            grapheme_spans=tuple(grapheme_spans),
            phoneme_spans=tuple(phoneme_spans),
        )

    @staticmethod
    def _convert_segment(segment: TokenizedSegment) -> str:
        if segment.is_chinese:
            return IPAConverter._convert_chinese(segment.raw)
        if segment.is_alpha:
            return IPAConverter._convert_english(segment.raw)
        return segment.raw

    @staticmethod
    def _sanitize_phoneme(value: str) -> str:
        without_markers = value.translate(_STRESS_TRANSLATION).translate(
            _TONE_TRANSLATION
        )
        without_whitespace = _WHITESPACE_RE.sub("", without_markers)
        return IPAConverter._remove_punctuation(without_whitespace)

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
