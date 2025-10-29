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


def test_ipa_converter_can_strip_markers(monkeypatch):
    monkeypatch.setattr(conversion, "eng_to_ipa_convert", lambda token: "ˈɛn ˌoʊ1")
    monkeypatch.setattr(conversion, "hanzi_to_ipa", lambda text: "pʰin1˥ ˈin2")

    converter = IPAConverter(
        remove_tone_marks=True, remove_stress_marks=True, strip_whitespace=True
    )

    result = converter.convert("Hello 世界 !")

    assert result == "ɛnoʊpʰinin!"


def test_ipa_converter_can_remove_punctuation():
    converter = IPAConverter(remove_punctuation=True)

    text = "Hello， 世界！ NASA?"

    assert converter.convert(text) == "en(Hello) zh(世界) en(N) en(A) en(S) en(A)"


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
    def fake_pw_align(sentence, query, mode):
        assert mode == "overlap"
        return (["s", "-", "‖", " ", "e"], ["s", " ", "e"], 0.75)

    monkeypatch.setattr(alignment, "pw_align", fake_pw_align)

    aligner = LocalAlignment(method="lingpy")
    result = aligner.align("sentence", "query")

    assert result.score == pytest.approx(0.75)
    assert result.sentence_alignment == "s-‖ e"
    assert result.query_alignment == "s e"
    assert result.sentence_subsequence == "se"


def test_local_alignment_with_aline_removes_markers(monkeypatch):
    class FakeAline:
        def __init__(self, mode):
            self.mode = mode

        def alignment(self, sentence, query):
            return (0.5, "s-‖ e", "q- e")

    fake_aline = FakeAline(mode="semi-global")

    aligner = LocalAlignment(method="aline", aline=fake_aline)
    result = aligner.align("sentence", "query")

    assert result.score == pytest.approx(0.5)
    assert result.sentence_alignment == "s-‖ e"
    assert result.query_alignment == "q- e"
    assert result.sentence_subsequence == "se"


def test_local_alignment_with_default_aline_uses_semi_global(monkeypatch):
    class FakeAline:
        def __init__(self, mode):
            self.mode = mode

        def alignment(self, sentence, query):
            return (0.0, "", "")

    monkeypatch.setattr(alignment, "ALINE", FakeAline)

    aligner = LocalAlignment(method="aline")

    assert isinstance(aligner.aline, FakeAline)
    assert aligner.aline.mode == "semi-global"


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
