"""Utilities for building and persisting IPA lexicons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from .conversion import GraphemePhoneme, IPAConverter

__all__ = ["IPALexicon"]


class IPALexicon:
    """Build and persist a lexicon of phrases and their IPA forms."""

    def __init__(self, converter: Optional[IPAConverter] = None) -> None:
        self.converter = converter or IPAConverter()
        self.entries: Dict[str, GraphemePhoneme] = {}

    def add_phrases(self, phrases: Iterable[Union[str, GraphemePhoneme]]) -> None:
        for phrase in phrases:
            if phrase is None:
                continue
            grapheme_phoneme = (
                phrase
                if isinstance(phrase, GraphemePhoneme)
                else GraphemePhoneme(str(phrase), converter=self.converter)
            )
            self.entries[grapheme_phoneme.graphemes] = grapheme_phoneme

    def save_to(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    graphemes: gp.phonemes
                    for graphemes, gp in self.entries.items()
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load_from(self, path: Path | str) -> None:
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        self.entries = {
            str(key): GraphemePhoneme.from_mapping(
                str(key), str(value), converter=self.converter
            )
            for key, value in data.items()
        }
