import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest

from asr_error_correction import IPAConverter, IPALexicon, LocalAlignment, local_align_sentence
import asr_error_correction.conversion as conversion
import asr_error_correction.alignment as alignment


@pytest.fixture(autouse=True)
def patch_converters(monkeypatch):
    monkeypatch.setattr(conversion, "eng_to_ipa_convert", lambda token: f"en({token})")
    monkeypatch.setattr(conversion, "hanzi_to_ipa", lambda text: f"zh({text})")


def test_ipa_converter_handles_mixed_languages():
    converter = IPAConverter()

    text = "Hello 世界 NASA!"
    expected = "en(Hello) zh(世界) en(N) en(A) en(S) en(A)!"

    assert converter.convert(text) == expected


def test_ipalexicon_save_and_load_roundtrip(tmp_path: Path):
    converter = IPAConverter()
    lexicon = IPALexicon(converter)
    lexicon.add_phrases(["Hello", "世界"])

    output_file = tmp_path / "lexicon.json"
    lexicon.save_to(output_file)

    loaded = IPALexicon(converter)
    loaded.load_from(output_file)

    assert loaded.entries == {
        "Hello": "en(Hello)",
        "世界": "zh(世界)",
    }


def test_local_alignment_with_lingpy(monkeypatch):
    def fake_we_align(sentence, query):
        return [(["s", "-", "e"], ["s", "e"], 0.75)]

    monkeypatch.setattr(alignment, "we_align", fake_we_align)

    aligner = LocalAlignment(method="lingpy")
    result = aligner.align("sentence", "query")

    assert result.score == pytest.approx(0.75)
    assert result.sentence_alignment == "s-e"
    assert result.query_alignment == "se"
    assert result.sentence_subsequence == "se"


def test_local_align_sentence_uses_aligner(monkeypatch):
    class DummyAligner:
        def __init__(self):
            self.calls = []

        def align(self, sentence, query):
            self.calls.append((sentence, query))
            return f"aligned-{query}"

    aligner = DummyAligner()
    results = local_align_sentence("sentence", ["one", "two"], aligner)

    assert aligner.calls == [("sentence", "one"), ("sentence", "two")]
    assert results == [("one", "aligned-one"), ("two", "aligned-two")]
