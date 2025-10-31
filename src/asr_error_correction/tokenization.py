"""Tokenisation helpers for mixed English/Chinese strings."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

__all__ = ["TokenizedSegment", "tokenize_mixed_text"]


_CHINESE_BLOCK = "\u3400-\u4dbf\u4e00-\u9fff"
_CHINESE_RE = re.compile(rf"[{_CHINESE_BLOCK}]+")
_TOKEN_RE = re.compile(rf"[{_CHINESE_BLOCK}]+|[A-Za-z]+|[^A-Za-z{_CHINESE_BLOCK}]+")


@dataclass(frozen=True)
class TokenizedSegment:
    """A token extracted from the input string prior to conversion."""

    raw: str
    start: int
    end: int

    @property
    def is_chinese(self) -> bool:
        """Return ``True`` when the token is comprised of Chinese characters."""

        return _CHINESE_RE.fullmatch(self.raw) is not None

    @property
    def is_alpha(self) -> bool:
        """Return ``True`` when the token contains only alphabetic characters."""

        return self.raw.isalpha()


def tokenize_mixed_text(text: str) -> Iterator[TokenizedSegment]:
    """Yield :class:`TokenizedSegment` instances detected within ``text``."""

    for match in _TOKEN_RE.finditer(text):
        yield TokenizedSegment(match.group(0), match.start(), match.end())

