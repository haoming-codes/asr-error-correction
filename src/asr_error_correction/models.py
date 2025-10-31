"""Core data models used across the ASR error correction package."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

__all__ = [
    "AlignmentResult",
    "GraphemePhoneme",
    "ReplacementPlan",
]


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

        grapheme_spans: list[tuple[int, int]] = []
        cursor = 0
        for grapheme in grapheme_list:
            index = grapheme_str.find(grapheme, cursor)
            if index == -1:
                raise ValueError("Unable to locate grapheme segment in grapheme_str")
            start = index
            end = index + len(grapheme)
            grapheme_spans.append((start, end))
            cursor = end

        phoneme_spans: list[tuple[int, int]] = []
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


@dataclass(frozen=True)
class AlignmentResult:
    """Description of a phonetic alignment between two grapheme/phoneme pairs."""

    score: float
    start: int
    end: int
    match: GraphemePhoneme

    def as_tuple(self) -> tuple[float, int, int, GraphemePhoneme]:
        """Return a tuple representation for backwards compatibility."""

        return (self.score, self.start, self.end, self.match)

    def __iter__(self) -> Iterator[object]:  # pragma: no cover - convenience iterator
        yield from self.as_tuple()


@dataclass(frozen=True)
class ReplacementPlan:
    """Description of a replacement within a sentence string."""

    score: float
    start: int
    end: int
    replacement: GraphemePhoneme

    def __iter__(self) -> Iterator[object]:  # pragma: no cover - convenience iterator
        yield from (self.score, self.start, self.end, self.replacement)

