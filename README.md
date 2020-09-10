# Phonetisaurus for Python

Python wrapper for the excellent [phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus) grapheme to phoneme tool ([license](https://github.com/AdolfVonKleist/Phonetisaurus/blob/master/LICENSE)).

Includes pre-built binaries for:

* `x86_64` - desktop/laptop/server (64-bit)
* `armv6l` - Raspberry Pi 0/1
* `armv7l` - Raspberry Pi 2/3/4 (32-bit)
* `armv8` - Raspberry Pi 3/4 (64-bit)

## Requirements

* Python 3.7+
* Linux

## Installing

```sh
$ pip install phonetisaurus
```

## Training

Assuming you have a lexicon formatted like [the CMU pronouncing dictionary](https://github.com/cmusphinx/cmudict):

```
word1 phoneme1 phoneme2 ...
word2 phoneme1 phoneme2 phoneme3 ...
```

saved to `lexicon.dict` run:

```sh
$ phonetisaurus train --model /path/to/write/g2p.fst /path/to/lexicon.dict
```

You may supply more than one lexicon.

See `phonetisaurus train --help` for more options.

## Predicting

```sh
$ phonetisaurus predict --model /path/to/g2p.fst word1 word2 ...
```

If no words are provided on the command line, they will be read line-by-line from standard in.

You may optionally supply one or more `--lexicon /path/to/lexicon.dict` arguments to avoid guessing pronunciations for known words.

See `phonetisaurus predict --help` for more options.
