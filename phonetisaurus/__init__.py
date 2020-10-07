"""Methods for training and using phonetisaurus"""
import gzip
import io
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import typing
from collections import defaultdict
from pathlib import Path

_LOGGER = logging.getLogger("phonetisaurus")

_DIR = Path(__file__).parent

# Excludes 0xA0
_WHITESPACE = re.compile(r"[ \t]+")

# word -> [[p1, p2], [p1, p2, p3]]
LEXICON_TYPE = typing.Dict[str, typing.List[typing.List[str]]]

# -----------------------------------------------------------------------------


def predict(
    words: typing.Iterable[str],
    model_path: typing.Union[str, Path],
    nbest: int = 1,
    env: typing.Optional[typing.Dict[str, str]] = None,
) -> typing.Iterable[typing.Tuple[str, typing.List[str]]]:
    """Guess one or more pronunciations for a set of words."""
    if env is None:
        env = guess_environment()

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as temp_file:
        # Write words to a temporary file
        for word in words:
            print(word, file=temp_file)

        # Rewind
        temp_file.seek(0)

        phonetisaurus_cmd = [
            "phonetisaurus-apply",
            "--model",
            shlex.quote(str(model_path)),
            "--word_list",
            shlex.quote(str(temp_file.name)),
            "--nbest",
            str(nbest),
        ]

        _LOGGER.debug(phonetisaurus_cmd)

        result_str: str = subprocess.check_output(
            phonetisaurus_cmd, env=env, universal_newlines=True
        )

        # Parse results
        with io.StringIO(result_str) as result_file:
            for line in result_file:
                line = line.strip()
                if line:
                    word, *phonemes = _WHITESPACE.split(line)
                    yield (word, phonemes)


# -----------------------------------------------------------------------------

# Skip lines with reserved symbols or 0xA0
_LEXICON_SKIP = re.compile(r".*[_|\xA0].*")


def train(
    lexicon: LEXICON_TYPE,
    model_path: typing.Union[str, Path],
    corpus_path: typing.Optional[typing.Union[str, Path]] = None,
    env: typing.Optional[typing.Dict[str, str]] = None,
):
    """Create a new grapheme to phoneme model based on a lexion"""
    # Create directories
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if corpus_path:
        corpus_path = Path(corpus_path)
        corpus_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Convert lexicon to appropriate format
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as temp_file:
            for word, word_prons in lexicon.items():
                for word_pron in word_prons:
                    word_pron_str = " ".join(word_pron)
                    lexicon_line = f"{word}\t{word_pron_str}"

                    if not _LEXICON_SKIP.match(lexicon_line):
                        print(lexicon_line, file=temp_file)

            # Rewind
            temp_file.seek(0)

            train_cmd = [
                "phonetisaurus-train",
                "--lexicon",
                shlex.quote(str(temp_file.name)),
                "--seq2_del",
                "--verbose",
            ]

            _LOGGER.debug(train_cmd)
            subprocess.check_call(train_cmd, cwd=temp_dir_str)

            model_fst = temp_dir / "train" / "model.fst"
            shutil.copy2(model_fst, model_path)

            if corpus_path:
                model_corpus = temp_dir / "train" / "model.corpus"
                shutil.copy2(model_corpus, corpus_path)


# -----------------------------------------------------------------------------

_WORD_WITH_NUMBER = re.compile(r"^([^(]+)(\(\d+\))$")


def load_lexicon(
    lexicon_file: typing.IO[str],
    word_separator: typing.Optional[str] = None,
    phoneme_separator: typing.Optional[str] = None,
    lexicon: typing.Optional[LEXICON_TYPE] = None,
) -> LEXICON_TYPE:
    """Load a CMU-style lexicon."""
    lexicon = lexicon or defaultdict(list)

    if word_separator:
        word_regex = re.compile(word_separator)
    else:
        word_regex = _WHITESPACE

    if phoneme_separator:
        phoneme_regex = re.compile(phoneme_separator)
    else:
        phoneme_regex = _WHITESPACE

    for line_idx, line in enumerate(lexicon_file):
        line = line.strip()
        if not line:
            continue

        try:
            word, phoneme_str = word_regex.split(line, maxsplit=1)
            phonemes = phoneme_regex.split(phoneme_str)

            word_match = _WORD_WITH_NUMBER.match(word)
            if word_match:
                # Strip (n) from word(n)
                word = word_match.group(1)

            word_prons = lexicon.get(word)
            if word_prons:
                word_prons.append(phonemes)
            else:
                lexicon[word] = [phonemes]
        except Exception:
            _LOGGER.exception("Line %s: %s", line_idx + 1, line)

    return lexicon


# -----------------------------------------------------------------------------


def guess_environment(machine: typing.Optional[str] = None) -> typing.Dict[str, str]:
    """Guess PATH and LD_LIBRARY_PATH based on machine type"""
    if not machine:
        machine = platform.machine()

    # Set bin/lib environment
    bin_dir = _DIR / "bin" / machine
    lib_dir = _DIR / "lib" / machine

    env = os.environ.copy()
    env = {
        "PATH": str(bin_dir) + ":" + env.get("PATH", ""),
        "LD_LIBRARY_PATH": str(lib_dir) + ":" + env.get("LD_LIBRARY", ""),
    }

    return env


# -----------------------------------------------------------------------------


def maybe_gzip_open(
    path_or_str: typing.Union[Path, str], mode: str = "r", create_dir: bool = True
) -> typing.IO[typing.Any]:
    """Opens a file as gzip if it has a .gz extension."""
    if create_dir and mode in {"w", "a"}:
        Path(path_or_str).parent.mkdir(parents=True, exist_ok=True)

    if str(path_or_str).endswith(".gz"):
        if mode == "r":
            gzip_mode = "rt"
        elif mode == "w":
            gzip_mode = "wt"
        elif mode == "a":
            gzip_mode = "at"
        else:
            gzip_mode = mode

        return gzip.open(path_or_str, gzip_mode)

    return open(path_or_str, mode)
