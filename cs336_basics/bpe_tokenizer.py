import json
from typing import Iterable, Iterator

from cs336_basics.bpe_training import pre_tokenize_string, split_on_special_tokens


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(self, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        """
        This method constructs the tokenizer from serialized vocabulary and merges files.
        @param vocab_filepath: Path to serialized vocabulary.
        @param merges_filepath: Path to serialized merges.
        @param special_tokens: Special tokens.
        """
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
            self.vocab = {int(token_id):token_str.encode('UTF-8') for token_str, token_id in raw_vocab.items()}

        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                part1, part2 = [part.encode("UTF-8") for part in line.split(" ", 1)]
                self.merges.append((part1, part2))

        for i, special_token in enumerate(special_tokens, start=len(self.vocab)):
            self.vocab[i] = special_token.encode("UTF-8")

    def encode(self, text: str) -> list[int]:
        """
        This method tokenizes the given text.
        @param text: Input text to be tokenized.
        @return tokenized_text: Tokenized text as list of integer IDs.
        """
        tokenized_text: list[int] = []
        pre_tokenized_text: list[tuple[bytes,...]] = []

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        splits = split_on_special_tokens(text, self.special_tokens, include_special=True)

        for split in splits:
            if self.special_tokens:
                if split in self.special_tokens:
                    pre_tokenized_text.append(tuple([split.encode("UTF-8")]))
                else:
                    pre_tokenized_text.extend(pre_tokenize_string(split, PAT))
            else:
                pre_tokenized_text.extend(pre_tokenize_string(split, PAT))

        for merge in self.merges:
            pre_tokenized_text = [apply_merge(pre_token, merge) for pre_token in pre_tokenized_text]

        vocab_inverse = {v:k for k,v in self.vocab.items()}
        for pre_token in pre_tokenized_text:
            tokenized_text.extend([vocab_inverse[token] for token in pre_token])

        return tokenized_text


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        This method tokenizes the given iterable of strings.
        @param iterable: Iterable containing strings to be tokenized.
        @return: Iterator over tokenIDs.
        """
        for string in iterable:
            yield from self.encode(string)


    def decode(self, ids: list[int]) -> str:
        """
        This method transforms the given integer IDs into their string representations.
        @param ids: List of integer IDs.
        @return: String representation of the given integer IDs.
        """
        result = b""
        for id in ids:
            result += self.vocab[id]
        return result.decode("UTF-8", errors='replace')


def apply_merge(pre_token: tuple[bytes,...], merge: tuple[bytes,bytes]) -> tuple[bytes,...]:
    """
    This method applies the given merges to the given pre_token.
    @param pre_token: Single pre-token for which the merges are applied.
    @param merge: Single merge to be applied.
    @return: Pre-token with all given merges applied.
    """
    new_pre_token: list[bytes] = []
    i = 0
    while i < len(pre_token):
        if i < len(pre_token) - 1 and (pre_token[i], pre_token[i + 1]) == merge:
            new_pre_token.append(merge[0] + merge[1])
            i += 2
        else:
            new_pre_token.append(pre_token[i])
            i += 1
    return tuple(new_pre_token)
