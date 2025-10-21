from __future__ import annotations
import json
import time
import os
import resource
import sys

import psutil
import pytest
import tiktoken

from .common import FIXTURES_PATH
from .adapters import run_train_bpe
from .adapters import get_tokenizer
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

'''
def test_train_bpe():
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    print('input_path', type(input_path), input_path)
    output_prefix = 'tinystories_sample_5M'
    with open('var/{}_merges.txt'.format(output_prefix), 'w') as fw:
        seq_str = '\n'.join([str(merge[0]) + ' ' + str(merge[1]) for merge in merges])
        fw.write(seq_str)
    with open('var/{}_vocab.json'.format(output_prefix), 'w') as fw:
        # tmp_vocab = {(k, v.decode('utf-8')) for k, v in vocab.items()}
        tmp_vocab = {str(k): str(v) for k, v in vocab.items()}
        json.dump(tmp_vocab, fw, indent=4)
'''


def test_unicode_string_with_special_tokens_matches_tiktoken():
    from .test_tokenizer import get_tokenizer_from_vocab_merges_path
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    print('\nreference_tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})',
          reference_tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    reference_ids = reference_tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


