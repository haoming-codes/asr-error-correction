"""Tools for converting mixed English/Chinese text into IPA."""
from __future__ import annotations

from decimal import Decimal
import re
import unicodedata
from typing import Iterable, Iterator, List, Tuple

from dragonmapper.hanzi import to_ipa as hanzi_to_ipa
from eng_to_ipa import convert as eng_to_ipa_convert

try:  # pragma: no cover - import is exercised indirectly in tests
    from num2words import num2words
except ImportError:  # pragma: no cover - dependency is optional at runtime
    num2words = None  # type: ignore[assignment]

from .models import GraphemePhoneme
from .tokenization import TokenizedSegment, tokenize_mixed_text


__all__ = ["GraphemePhoneme", "IPAConverter", "TokenizedSegment"]


_STRESS_TRANSLATION = str.maketrans("", "", "ˈˌ")
_TONE_TRANSLATION = str.maketrans("", "", "012345˥˦˧˨˩˩˨˧˦˥")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


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
            for value, is_ipa_segment in self._iter_converted_values(token):
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
            else:
                for start, end, phoneme in self._iter_numeric_phonemes(token):
                    if phoneme is None:
                        continue
                    phoneme = self._apply_marker_options(phoneme)
                    sanitized = self._sanitize_phoneme(phoneme)
                    if not sanitized:
                        continue
                    grapheme_list.append(text[start:end])
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

    def _iter_converted_values(
        self, segment: TokenizedSegment
    ) -> Iterator[Tuple[str, bool]]:
        if segment.is_chinese:
            yield IPAConverter._convert_chinese(segment.raw), True
            return
        if segment.is_alpha:
            yield IPAConverter._convert_english(segment.raw), True
            return

        last = 0
        for match in _NUMBER_RE.finditer(segment.raw):
            start, end = match.span()
            if start > last:
                yield segment.raw[last:start], False
            number_value = self._convert_number(match.group(0))
            if number_value is None:
                yield segment.raw[start:end], False
            else:
                yield number_value, True
            last = end

        if last < len(segment.raw):
            yield segment.raw[last:], False

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

    def _convert_number(self, number: str) -> str | None:
        if not self.num2words_lang:
            return None
        if num2words is None:  # pragma: no cover - defensive branch
            raise RuntimeError(
                "num2words is required when num2words_lang is configured"
            )

        if "." in number:
            value: int | Decimal = Decimal(number)
        else:
            value = int(number)

        words = num2words(value, lang=self.num2words_lang)
        if self.num2words_lang.lower().startswith("zh"):
            return IPAConverter._convert_chinese(words)
        return IPAConverter._convert_english(words)

    def _iter_numeric_phonemes(
        self, segment: TokenizedSegment
    ) -> Iterator[Tuple[int, int, str | None]]:
        for match in _NUMBER_RE.finditer(segment.raw):
            start, end = match.span()
            yield segment.start + start, segment.start + end, self._convert_number(
                match.group(0)
            )
