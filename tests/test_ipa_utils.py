import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

import ipa_utils


@pytest.fixture(autouse=True)
def patch_converters(monkeypatch):
    monkeypatch.setattr(ipa_utils, "eng_to_ipa_convert", lambda token: f"en({token})")
    monkeypatch.setattr(ipa_utils, "hanzi_to_ipa", lambda text: f"zh({text})")


def test_ipa_converter_handles_mixed_languages():
    converter = ipa_utils.IPAConverter()

    text = "Hello 世界 NASA!"
    expected = "en(Hello) zh(世界) en(N) en(A) en(S) en(A)!"

    assert converter.convert(text) == expected


def test_ipalexicon_save_and_load_roundtrip(tmp_path: Path):
    converter = ipa_utils.IPAConverter()
    lexicon = ipa_utils.IPALexicon(converter)
    lexicon.add_phrases(["Hello", "世界"])

    output_file = tmp_path / "lexicon.json"
    lexicon.save_to(output_file)

    loaded = ipa_utils.IPALexicon(converter)
    loaded.load_from(output_file)

    assert loaded.entries == {
        "Hello": "en(Hello)",
        "世界": "zh(世界)",
    }


def test_local_alignment_with_lingpy(monkeypatch):
    def fake_we_align(sentence, query):
        return [(["s", "-", "e"], ["s", "e"], 0.75)]

    monkeypatch.setattr(ipa_utils, "we_align", fake_we_align)

    aligner = ipa_utils.LocalAlignment(method="lingpy")
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
    results = ipa_utils.local_align_sentence("sentence", ["one", "two"], aligner)

    assert aligner.calls == [("sentence", "one"), ("sentence", "two")]
    assert results == [("one", "aligned-one"), ("two", "aligned-two")]
