"""Tools for converting mixed English/Chinese text into IPA."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Tuple

from dragonmapper.hanzi import to_ipa as hanzi_to_ipa
from eng_to_ipa import convert as eng_to_ipa_convert
try:
    from num2words import num2words
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    raise ModuleNotFoundError(
        "The 'num2words' package is required for numerical conversion. "
        "Install it with 'pip install num2words'."
    ) from exc

from .models import GraphemePhoneme
from .tokenization import TokenizedSegment, tokenize_mixed_text


__all__ = ["GraphemePhoneme", "IPAConverter", "TokenizedSegment"]


_STRESS_TRANSLATION = str.maketrans("", "", "ˈˌ")
_TONE_TRANSLATION = str.maketrans("", "", "012345˥˦˧˨˩˩˨˧˦˥")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"(?<!\w)([+-]?\d[\d,]*(?:\.\d+)?)(?!\w)")


class IPAConverter:
    """Convert English and Chinese text to their IPA representations."""

    def __init__(
        self,
        *,
        remove_tone_marks: bool = False,
        remove_stress_marks: bool = False,
        strip_whitespace: bool = False,
        remove_punctuation: bool = False,
        num2words_lang: str | None = None,
    ) -> None:
        self.remove_tone_marks = remove_tone_marks
        self.remove_stress_marks = remove_stress_marks
        self.strip_whitespace = strip_whitespace
        self.remove_punctuation = remove_punctuation
        self.num2words_lang = num2words_lang

    def tokenize(self, text: str) -> Iterable[TokenizedSegment]:
        """Yield the detected segments from ``text``."""
        yield from tokenize_mixed_text(text)

    def convert(self, text: str) -> str:
        """Convert a text containing English and Chinese characters to IPA."""
        if not text:
            return ""

        converted: List[Tuple[str, bool]] = []
        for token in self.tokenize(text):
            value, is_ipa_segment = self._convert_segment(token)
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
            elif self.num2words_lang:
                for match in _NUMBER_RE.finditer(token.raw):
                    grapheme = match.group(1)
                    phoneme = self._convert_number_string_to_ipa(grapheme)
                    phoneme = self._apply_marker_options(phoneme)
                    sanitized = self._sanitize_phoneme(phoneme)
                    if not sanitized:
                        continue
                    start = token.start + match.start(1)
                    end = token.start + match.end(1)
                    grapheme_list.append(grapheme)
                    grapheme_spans.append((start, end))
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

    def _convert_segment(self, segment: TokenizedSegment) -> Tuple[str, bool]:
        if segment.is_chinese:
            return self._convert_chinese(segment.raw), True
        if segment.is_alpha:
            return self._convert_english(segment.raw), True
        if self.num2words_lang:
            converted, replaced = self._replace_numbers_with_ipa(segment.raw)
            return converted, replaced
        return segment.raw, False

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

    def _convert_number_string_to_ipa(self, value: str) -> str:
        words = num2words(value, lang=self.num2words_lang)
        if self.num2words_lang and self.num2words_lang.startswith("zh"):
            return self._convert_chinese(words)
        return self._convert_english(words)

    def _replace_numbers_with_ipa(self, text: str) -> Tuple[str, bool]:
        replaced = False

        def repl(match: re.Match[str]) -> str:
            nonlocal replaced
            replaced = True
            number_text = match.group(1)
            return self._convert_number_string_to_ipa(number_text)

        converted = _NUMBER_RE.sub(repl, text)
        return converted, replaced

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
