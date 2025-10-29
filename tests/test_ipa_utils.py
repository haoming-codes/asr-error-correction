import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest

from asr_error_correction import (
    GraphemePhoneme,
    IPAConverter,
    IPALexicon,
    LocalAlignment,
    local_align_sentence,
)
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

    hello = loaded.entries["Hello"]
    assert hello.grapheme_list == ("Hello",)
    assert hello.phoneme_list == ("enHello",)

    world = loaded.entries["世界"]
    assert world.grapheme_list == ("世", "界")
    assert world.phoneme_list == ("zh世", "zh界")


def test_converter_builds_grapheme_phoneme_structure():
    converter = IPAConverter()

    gp = converter.convert_to_grapheme_phoneme("你好 Hello 世界!")

    assert gp.grapheme_list == ("你", "好", "Hello", "世", "界")
    assert gp.phoneme_list == ("zh你", "zh好", "enHello", "zh世", "zh界")
    assert gp.phoneme_str == "zh你zh好enHellozh世zh界"


def _build_sentence_and_query() -> tuple[GraphemePhoneme, GraphemePhoneme]:
    sentence = GraphemePhoneme.from_components(
        "你好 hello world",
        ["你", "好", "hello", "world"],
        "nixɑʊhɛloʊwərld",
        ["ni", "xɑʊ", "hɛloʊ", "wərld"],
    )
    query = GraphemePhoneme.from_components(
        "好哈喽",
        ["好", "哈", "喽"],
        "xɑʊhɑloʊ",
        ["xɑʊ", "hɑ", "loʊ"],
    )
    return sentence, query


def test_local_alignment_with_lingpy(monkeypatch):
    sentence, query = _build_sentence_and_query()

    def fake_pw_align(sentence, query, *, mode):
        assert mode == "overlap"
        return (
            list(sentence),
            ["-", "-", "x", "ɑ", "ʊ", "h", "ɑ", "l", "o", "ʊ"],
            0.75,
        )

    monkeypatch.setattr(alignment, "pw_align", fake_pw_align)

    aligner = LocalAlignment()
    result = aligner.align(sentence, query)

    assert result[0][0] == pytest.approx(0.75)
    matched_gp = result[0][1]
    assert matched_gp.grapheme_list == ("好", "hello")
    assert matched_gp.phoneme_list == ("xɑʊ", "hɛloʊ")
    assert matched_gp.grapheme_str == "好 hello"
    assert matched_gp.phoneme_str == "xɑʊhɛloʊ"


def test_local_align_sentence_uses_aligner(monkeypatch):
    sentence, query_one = _build_sentence_and_query()
    query_two = GraphemePhoneme.from_components(
        "你好",
        ["你", "好"],
        "nixɑʊ",
        ["ni", "xɑʊ"],
    )

    class DummyAligner:
        def __init__(self):
            self.calls = []

        def align(self, sentence, query):
            self.calls.append((sentence, query))
            return [(1.0, query)]

    aligner = DummyAligner()
    results = local_align_sentence(sentence, [query_one, query_two], aligner)

    assert aligner.calls == [(sentence, query_one), (sentence, query_two)]
    assert results == [
        (query_one, [(1.0, query_one)]),
        (query_two, [(1.0, query_two)]),
    ]


def test_local_alignment_filters_marker_only_matches(monkeypatch):
    def fake_pw_align(sentence, query, *, mode):
        return (["-", "‖", " "], ["-", "‖", " "], 1.2)

    monkeypatch.setattr(alignment, "pw_align", fake_pw_align)

    aligner = LocalAlignment()

    sentence, query = _build_sentence_and_query()

    assert aligner.align(sentence, query) == []
