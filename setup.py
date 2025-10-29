from pathlib import Path

from setuptools import find_packages, setup

README = Path(__file__).parent / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

setup(
    name="asr-error-correction",
    version="0.1.0",
    description="Utilities for converting mixed-language text to IPA and aligning pronunciations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://example.com/asr-error-correction",
    author="",
    packages=find_packages(exclude=("tests",)),
    py_modules=["ipa_utils"],
    install_requires=[
        "pypinyin",
        "panphon",
        "lingpy",
        "eng_to_ipa",
        "dragonmapper",
        "sktime",
        "openai",
        "abydos @ git+https://github.com/denizberkin/abydos.git",
    ],
    python_requires=">=3.8",
)
