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
    AlignmentResult,
    ASRCorrector,
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

    mapping = {
        ("en", "3"): "three",
        ("en", "3.14"): "three point one four",
        ("en", "361"): "three hundred sixty-one",
        ("zh_CN", "361"): "三百六十一",
    }

    def fake_num2words(value, *, lang):
        key = (lang, str(value))
        if key not in mapping:
            raise AssertionError(f"Unexpected num2words request: {key}")
        return mapping[key]

    monkeypatch.setattr(conversion, "num2words", fake_num2words)


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


def test_ipa_converter_converts_numbers_to_words():
    converter = IPAConverter(num2words_lang="en")

    text = "Call 361 now."

    assert (
        converter.convert(text)
        == "en(Call) en(three hundred sixty-one) en(now)."
    )


def test_ipa_converter_handles_decimal_numbers():
    converter = IPAConverter(num2words_lang="en")

    text = "Pi is about 3.14. Also 3."

    assert (
        converter.convert(text)
        == "en(Pi) en(is) en(about) en(three point one four). en(Also) en(three)."
    )


def test_ipa_converter_converts_numbers_in_chinese():
    converter = IPAConverter(num2words_lang="zh_CN")

    text = "今天361"

    assert converter.convert(text) == "zh(今天)zh(三百六十一)"


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


def test_converter_grapheme_phoneme_includes_numbers():
    converter = IPAConverter(num2words_lang="en")

    gp = converter.convert_to_grapheme_phoneme("Call 361")

    assert gp.grapheme_list == ("Call", "361")
    assert gp.phoneme_list == ("enCall", "enthreehundredsixtyone")
    assert gp.phoneme_str == "enCallenthreehundredsixtyone"


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

    assert result[0].score == pytest.approx(0.75)
    assert result[0].start == 1
    assert result[0].end == 8
    matched_gp = result[0].match
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
            return [
                AlignmentResult(
                    score=1.0,
                    start=0,
                    end=len(query.grapheme_str),
                    match=query,
                )
            ]

    aligner = DummyAligner()
    results = local_align_sentence(sentence, [query_one, query_two], aligner)

    assert aligner.calls == [(sentence, query_one), (sentence, query_two)]
    assert results[0][0] is query_one
    assert results[0][1][0].score == 1.0
    assert results[0][1][0].end == len(query_one.grapheme_str)
    assert results[0][1][0].match is query_one

    assert results[1][0] is query_two
    assert results[1][1][0].score == 1.0
    assert results[1][1][0].end == len(query_two.grapheme_str)
    assert results[1][1][0].match is query_two


def _build_sentence_gp() -> GraphemePhoneme:
    return GraphemePhoneme.from_components(
        "你好 hello world",
        ["你", "好", "hello", "world"],
        "nixɑʊhɛloʊwərld",
        ["ni", "xɑʊ", "hɛloʊ", "wərld"],
    )


def test_asr_corrector_replaces_highest_scoring_match(monkeypatch):
    sentence_gp = _build_sentence_gp()

    match = sentence_gp.subsequence_covering_span(2, 10)
    assert match is not None

    replacement_entry = GraphemePhoneme.from_components(
        "号halo",
        ["号", "halo"],
        "xɑhalo",
        ["xɑ", "halo"],
    )

    lower_entry = GraphemePhoneme.from_components(
        "你好",
        ["你", "好"],
        "nixɑʊ",
        ["ni", "xɑʊ"],
    )

    class DummyConverter:
        def convert_to_grapheme_phoneme(self, text: str) -> GraphemePhoneme:
            assert text == "你好 hello world"
            return sentence_gp

    class DummyAligner:
        def align(self, sentence, query):
            if query is replacement_entry:
                return [AlignmentResult(score=0.9, start=1, end=8, match=match)]
            if query is lower_entry:
                lower_match = sentence.subsequence_covering_span(0, 2)
                assert lower_match is not None
                return [
                    AlignmentResult(score=0.5, start=0, end=2, match=lower_match)
                ]
            return []

    lexicon = IPALexicon()
    lexicon.converter = DummyConverter()
    lexicon.entries = {
        "号halo": replacement_entry,
        "你好": lower_entry,
    }

    corrector = ASRCorrector(lexicon, aligner=DummyAligner(), score_threshold=0.4)

    corrected = corrector.correct("你好 hello world")

    assert corrected == "你号halo world"


def test_asr_corrector_respects_score_threshold(monkeypatch):
    sentence_gp = _build_sentence_gp()

    class DummyConverter:
        def convert_to_grapheme_phoneme(self, text: str) -> GraphemePhoneme:
            return sentence_gp

    class DummyAligner:
        def align(self, sentence, query):
            match = sentence.subsequence_covering_span(2, 10)
            assert match is not None
            return [AlignmentResult(score=0.3, start=1, end=8, match=match)]

    lexicon = IPALexicon()
    lexicon.converter = DummyConverter()
    lexicon.entries = {
        "号halo": GraphemePhoneme.from_components(
            "号halo",
            ["号", "halo"],
            "xɑhalo",
            ["xɑ", "halo"],
        )
    }

    corrector = ASRCorrector(lexicon, aligner=DummyAligner(), score_threshold=0.5)

    assert corrector.correct("你好 hello world") == "你好 hello world"


def test_local_alignment_filters_marker_only_matches(monkeypatch):
    def fake_pw_align(sentence, query, *, mode):
        return (["-", "‖", " "], ["-", "‖", " "], 1.2)

    monkeypatch.setattr(alignment, "pw_align", fake_pw_align)

    aligner = LocalAlignment()

    sentence, query = _build_sentence_and_query()

    assert aligner.align(sentence, query) == []
