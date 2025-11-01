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
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pypinyin",
        "panphon",
        "lingpy",
        "eng_to_ipa",
        "dragonmapper",
        "sktime",
        "openai",
        "tqdm",
        "pandas",
        "python-calamine",
        "abydos @ git+https://github.com/denizberkin/abydos.git",
        "num2words @ git+https://github.com/savoirfairelinux/num2words.git@master"
    ],
    python_requires=">=3.8",
)
