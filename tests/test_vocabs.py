import pandas as pd

from src.data.vocab import Vocab


def test_vocab_unknown_lookup():
    series = pd.Series(["a", "b", "a", "c"])
    vocab = Vocab.from_series(series, min_freq=2, unknown_token="UNKNOWN")
    assert vocab.lookup("a") == vocab.stoi["a"]
    assert vocab.lookup("nonexistent") == vocab.stoi["UNKNOWN"]


def test_vocab_serialization_roundtrip():
    series = pd.Series(["x", "y", "y"])
    vocab = Vocab.from_series(series, min_freq=1, unknown_token="UNKNOWN")
    restored = Vocab.from_dict(vocab.to_dict())
    assert restored.stoi == vocab.stoi
    assert restored.itos == vocab.itos
