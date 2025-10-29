"""Tools for converting mixed English/Chinese text into IPA."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

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
    start: int
    end: int

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
        for match in _TOKEN_RE.finditer(text):
            yield TokenizedSegment(match.group(0), match.start(), match.end())

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
@dataclass(frozen=True)
class GraphemePhoneme:
    """Container storing aligned grapheme/phoneme data for a phrase."""

    grapheme_str: str
    grapheme_list: Tuple[str, ...]
    phoneme_str: str
    phoneme_list: Tuple[str, ...]
    grapheme_spans: Tuple[Tuple[int, int], ...]
    phoneme_spans: Tuple[Tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.grapheme_list) != len(self.phoneme_list):
            raise ValueError("grapheme_list and phoneme_list must be the same length")
        if len(self.grapheme_spans) != len(self.grapheme_list):
            raise ValueError("grapheme_spans must align with grapheme_list")
        if len(self.phoneme_spans) != len(self.phoneme_list):
            raise ValueError("phoneme_spans must align with phoneme_list")
        joined = "".join(self.phoneme_list)
        if joined != self.phoneme_str:
            raise ValueError("phoneme_str must be the concatenation of phoneme_list")

    def subsequence_covering_span(self, start: int, end: int) -> "GraphemePhoneme" | None:
        """Return a new object covering ``start``..``end`` in ``phoneme_str``.

        The returned object expands the ``start``/``end`` span to align with the
        full grapheme/phoneme tokens that overlap with the requested region. If
        no tokens overlap with the span ``None`` is returned.
        """

        if start >= end:
            return None

        indices = [
            idx
            for idx, (token_start, token_end) in enumerate(self.phoneme_spans)
            if token_end > start and token_start < end
        ]
        if not indices:
            return None

        first_idx, last_idx = indices[0], indices[-1]
        first_graph_start, _ = self.grapheme_spans[first_idx]
        _, last_graph_end = self.grapheme_spans[last_idx]

        sub_grapheme_str = self.grapheme_str[first_graph_start:last_graph_end]
        sub_grapheme_list = self.grapheme_list[first_idx : last_idx + 1]
        sub_phoneme_list = self.phoneme_list[first_idx : last_idx + 1]

        sub_grapheme_spans = tuple(
            (start_ - first_graph_start, end_ - first_graph_start)
            for (start_, end_) in self.grapheme_spans[first_idx : last_idx + 1]
        )

        phoneme_position = 0
        sub_phoneme_spans = []
        for value in sub_phoneme_list:
            span_end = phoneme_position + len(value)
            sub_phoneme_spans.append((phoneme_position, span_end))
            phoneme_position = span_end

        return GraphemePhoneme(
            grapheme_str=sub_grapheme_str,
            grapheme_list=sub_grapheme_list,
            phoneme_str="".join(sub_phoneme_list),
            phoneme_list=sub_phoneme_list,
            grapheme_spans=sub_grapheme_spans,
            phoneme_spans=tuple(sub_phoneme_spans),
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the object."""

        return {
            "grapheme_str": self.grapheme_str,
            "grapheme_list": list(self.grapheme_list),
            "phoneme_str": self.phoneme_str,
            "phoneme_list": list(self.phoneme_list),
        }

    @classmethod
    def from_components(
        cls,
        grapheme_str: str,
        grapheme_list: Sequence[str],
        phoneme_str: str,
        phoneme_list: Sequence[str],
    ) -> "GraphemePhoneme":
        """Build an instance from stored components."""

        grapheme_spans: List[Tuple[int, int]] = []
        cursor = 0
        for grapheme in grapheme_list:
            index = grapheme_str.find(grapheme, cursor)
            if index == -1:
                raise ValueError("Unable to locate grapheme segment in grapheme_str")
            start = index
            end = index + len(grapheme)
            grapheme_spans.append((start, end))
            cursor = end

        phoneme_spans: List[Tuple[int, int]] = []
        position = 0
        for phoneme in phoneme_list:
            end = position + len(phoneme)
            phoneme_spans.append((position, end))
            position = end

        return cls(
            grapheme_str=grapheme_str,
            grapheme_list=tuple(grapheme_list),
            phoneme_str=phoneme_str,
            phoneme_list=tuple(phoneme_list),
            grapheme_spans=tuple(grapheme_spans),
            phoneme_spans=tuple(phoneme_spans),
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "GraphemePhoneme":
        """Construct an instance from ``payload`` produced by :meth:`to_dict`."""

        return cls.from_components(
            payload["grapheme_str"],
            payload["grapheme_list"],
            payload["phoneme_str"],
            payload["phoneme_list"],
        )
