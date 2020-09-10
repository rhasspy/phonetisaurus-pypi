#!/usr/bin/env python3
"""Friendlier command-line interface for phonetisaurus"""
import argparse
import logging
import os
import platform
import sys
import time
from pathlib import Path

from . import load_lexicon, maybe_gzip_open, predict, train

_DIR = Path(__file__).parent

_LOGGER = logging.getLogger("phonetisaurus")

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.machine:
        args.machine = platform.machine()

    args.model = Path(args.model)

    _LOGGER.debug(args)

    # Set bin/lib environment
    bin_dir = _DIR / "bin" / args.machine
    lib_dir = _DIR / "lib" / args.machine
    _LOGGER.debug("bin=%s, lib=%s", bin_dir, lib_dir)

    env = os.environ.copy()
    env = {
        "PATH": str(bin_dir) + ":" + env.get("PATH", ""),
        "LD_LIBRARY_PATH": str(lib_dir) + ":" + env.get("LD_LIBRARY", ""),
    }

    # Set word casing
    casing = None
    if args.casing == "lower":
        casing = str.lower
    elif args.casing == "upper":
        casing = str.upper

    # Run command
    if args.command == "predict":
        # Predict pronunciations

        # Load optional lexicons
        args.lexicon = [Path(lexicon) for lexicon in args.lexicon]
        lexicon = None
        for lexicon_path in args.lexicon:
            if lexicon_path.is_file():
                _LOGGER.debug("Loading lexicon from %s", lexicon_path)
                with maybe_gzip_open(lexicon_path, "r") as lexicon_file:
                    lexicon = load_lexicon(lexicon_file, lexicon=lexicon)

        if lexicon:
            _LOGGER.debug("Loaded pronunciations for %s word(s)", len(lexicon))

        # Get words from arguments or stdin
        if args.words:
            words = args.words
        else:
            if os.isatty(sys.stdin.fileno()):
                print("Reading words from stdin. CTRL+D to end.", file=sys.stderr)

            words = []
            for word in sys.stdin:
                word = word.strip()
                if casing:
                    word = casing(word)

                words.append(word)

        if lexicon:
            # Try to look up first in the lexicon
            unknown_words = []
            num_known_words = 0
            for word in words:
                word_prons = lexicon.get(word)
                if word_prons:
                    for word_pron in word_prons[: args.nbest]:
                        pron_str = " ".join(word_pron)
                        print(word, pron_str, sep=args.word_separator)
                        num_known_words += 1
                else:
                    # Will need to guess
                    unknown_words.append(word)

            if num_known_words > 0:
                _LOGGER.debug("Looked up %s/%s word(s)", num_known_words, len(words))

            # Remaining words will be guessed
            words = unknown_words

        # Guess pronunciations
        if words:
            _LOGGER.debug("Guessing pronunciations for %s word(s)", len(words))
            for word, pron_str in predict(
                words=words, model_path=args.model, nbest=args.nbest, env=env
            ):
                print(word, pron_str, sep=args.word_separator)
    elif args.command == "train":
        # Train new model
        if args.corpus:
            args.corpus = Path(args.corpus)

        # Load lexicons
        args.lexicon = [Path(lexicon) for lexicon in args.lexicon]
        lexicon = None
        for lexicon_path in args.lexicon:
            if lexicon_path.is_file():
                _LOGGER.debug("Loading lexicon from %s", lexicon_path)
                with maybe_gzip_open(lexicon_path, "r") as lexicon_file:
                    lexicon = load_lexicon(lexicon_file, lexicon=lexicon)

        _LOGGER.debug("Loaded pronunciations for %s word(s)", len(lexicon))

        _LOGGER.debug("Started training")
        start_time = time.perf_counter()

        train(lexicon=lexicon, model_path=args.model, corpus_path=args.corpus, env=env)

        end_time = time.perf_counter()
        _LOGGER.debug("Finished training in %s second(s)", end_time - start_time)


# -----------------------------------------------------------------------------


def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="phonetisaurus")

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # -------
    # predict
    # -------
    predict_parser = sub_parsers.add_parser(
        "predict", help="Predict one or more pronunciations from words"
    )
    predict_parser.add_argument("words", nargs="*", help="Words to predict")
    predict_parser.add_argument(
        "--lexicon",
        default=[],
        action="append",
        help="Optional lexicon(s) to consult before guessing pronuncation(s)",
    )
    predict_parser.add_argument(
        "--nbest",
        type=int,
        default=1,
        help="Number of pronunciations to predict per word",
    )
    predict_parser.add_argument(
        "--word-separator",
        default=" ",
        help="Separator between words and pronunciations (default: space)",
    )

    # -------
    # train
    # -------
    train_parser = sub_parsers.add_parser(
        "train",
        help="Train a new model from one or more lexicons (phonetic dictionaries)",
    )
    train_parser.add_argument(
        "lexicon", nargs="+", help="Path(s) to read one or more phonetic dictionaries"
    )
    train_parser.add_argument("--corpus", help="Path to write trained g2p corpus")
    train_parser.add_argument(
        "--word-separator",
        default="\\s+",
        help="Separator regex between words in each lexicon entry (default: \\s+)",
    )
    train_parser.add_argument(
        "--phoneme-separator",
        default="\\s+",
        help="Separator regex between phonemes in each lexicon entry (default: \\s+)",
    )

    # Shared arguments
    for sub_parser in [predict_parser, train_parser]:
        sub_parser.add_argument("--model", required=True, help="Path to g2p model")
        sub_parser.add_argument(
            "--casing",
            choices=["lower", "upper", "ignore"],
            default="ignore",
            help="Case transformation to apply to words",
        )

        sub_parser.add_argument(
            "--debug", action="store_true", help="Print DEBUG messages to the console"
        )

        sub_parser.add_argument(
            "--machine",
            choices=["x86_64", "armv6l", "armv7l", "armv8"],
            help="Override detected platform machine type",
        )

    return parser.parse_args()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
