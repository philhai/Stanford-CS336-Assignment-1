import collections
from collections import defaultdict
import regex as re
import multiprocessing.pool
from cs336_basics.pretokenization_example import find_chunk_boundaries


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_workers=8) -> (
        dict[bytes, int], list[tuple[bytes, bytes]]):
    """
    This method trains a bpe tokenizer on a given input file.
    @param input_path: Path to input file.
    @param vocab_size: Desired vocabulary size after merges.
    @param special_tokens: Special tokens to be added to the vocabulary.
    @param num_workers: Number of parallel processes.
    @return (vocab, merges): Final vocabulary and the list of merges performed by the tokenizer.
    """
    vocab: dict[int, bytes] = {i:i.to_bytes(1, "big") for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    pre_token_counts: dict[tuple[bytes,...], int] = collections.Counter()

    for i, special_token in enumerate(special_tokens, start=len(vocab)):
        vocab[i] = special_token.encode("UTF-8")

    pool, worker_tasks = setup_multiprocessing(input_path, num_workers, special_tokens)

    for chunk_pre_token_counts in pool.starmap(pre_tokenize_chunk, worker_tasks):
        pre_token_counts.update(chunk_pre_token_counts)

    token_pair_counts = count_token_pairs(pre_token_counts)
    while len(vocab) < vocab_size:
        new_token, merge, pre_token_counts = merge_most_freq_token_pair(token_pair_counts, pre_token_counts)

        vocab[len(vocab)] = new_token
        merges.append(merge)

    return vocab, merges


def setup_multiprocessing(input_path: str, num_workers: int, special_tokens: list[str]) -> \
        (multiprocessing.Pool, list[tuple[str,int,int,list[str]]]):
    """
    This method creates a multiprocessing pool and the worker tasks to be mapped to the pool.
    @param input_path: Path to input file.
    @param num_workers: Number of parallel processes.
    @param special_tokens: List of special tokens.
    @return (pool, worker_tasks): Multiprocessing pool and the worker tasks to be mapped to the pool.
    """
    pool = multiprocessing.Pool(processes=num_workers)
    chunk_boundaries = find_chunk_boundaries(open(input_path, 'rb'), num_workers, b"<|endoftext|>")
    worker_tasks = [(input_path, start, end, special_tokens) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]
    return pool, worker_tasks


def pre_tokenize_chunk(input_path: str, start: int, end: int, special_tokens:list[str]) -> dict[tuple[bytes,...], int]:
    """
    This method implements a regex based pre-tokenizer.
    @param input_path: Path to the input file.
    @param start: Start of chunk boundary.
    @param end: End of chunk boundary.
    @param special_tokens: List of special tokens.
    @return pre_token_counts: A dictionary mapping from pre-tokens to their count. The keys are all pre-tokens represented as byte strings.
    """
    chunk = read_chunk(input_path, start, end)

    chunk_splits = split_on_special_tokens(chunk, special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_token_counts: dict[tuple[bytes,...], int] = collections.Counter()

    for split in chunk_splits:
        pre_tokenizer_result = pre_tokenize_string(split, PAT)
        pre_token_counts.update(collections.Counter(pre_tokenizer_result))

    return pre_token_counts


def read_chunk(input_path: str, start: int, end: int) -> str:
    """
    This method reads a chunk (start - end) of the given input file.
    @param input_path: Input file path.
    @param start: Start of chunk boundary.
    @param end: End of chunk boundary.
    @return chunk: Chunk of input file as UTF-8 string.
    """
    input_file = open(input_path, 'rb')
    input_file.seek(start)
    binary_chunk = input_file.read(end - start)
    chunk = binary_chunk.decode("utf-8", errors="ignore")
    return chunk


def split_on_special_tokens(chunk: str, special_tokens: list[str], include_special: bool = False) -> list[str]:
    """
    This method splits a given chunk into pieces according to the special tokens.
    @param chunk: Chunk of text to be split.
    @param special_tokens: List of special tokens.
    @param include_special: Boolean indicating whether the special tokens should be included in the result.
    @return chunk_splits: List of chunk splits.
    """
    if not special_tokens:
        return [chunk]
    special_tokens.sort(reverse=True)
    split_pattern = "|".join(map(re.escape, special_tokens))
    if include_special:
        split_pattern = "(" + split_pattern +  ")"
    chunk_splits = re.split(split_pattern, chunk)
    return [chunk_split for chunk_split in chunk_splits if chunk_split]


def pre_tokenize_string(input_string: str, PAT: str) -> list[tuple[bytes,...]]:
    """
    This method pre-tokenizes a given string according to the given RegEx.
    @param input_string: Input string to be pre-tokenized.
    @param PAT: Regex for pre-tokenization.
    @return encoded_pre_tokens_byte_tuples:  A list of pre-tokens where each pre-token is represented as a tuple of
    single bytes.
    """
    pre_tokens = re.finditer(PAT, input_string)
    encoded_pre_tokens = [pre_token.group().encode("UTF-8") for pre_token in pre_tokens]
    encoded_pre_tokens_byte_tuples = [bytes_to_tuple_of_bytes(x) for x in encoded_pre_tokens]
    return encoded_pre_tokens_byte_tuples


def bytes_to_tuple_of_bytes(input_bytes: bytes) -> tuple[bytes, ...]:
    """
    This method converts an input byte string to a tuple of single bytes. Each element in the output tuple is a bytes
    object representing a single byte.
    @param input_bytes: Input byte string.
    @return: A tuple where each element is a bytes object representing a single byte.
    """
    return tuple(input_bytes[i:i+1] for i in range(len(input_bytes)))


def count_token_pairs(pre_token_counts: dict[tuple[bytes,...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    This method sums up all consecutive token-pairs.
    @param pre_token_counts: A dictionary mapping from pre-tokens to their count. The keys are all pre-tokens represented as byte strings.
    @return token_pair_counts: A dictionary mapping from token pairs (represented as tuple[bytes]) to their count.
    """
    token_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for pre_token, count in pre_token_counts.items():
        for a, b in zip(pre_token[:-1], pre_token[1:]):
            token_pair_counts[(a, b)] += count
    return token_pair_counts


def merge_most_freq_token_pair(token_pair_counts: dict[tuple[bytes, bytes], int], pre_token_counts: dict[tuple[bytes,...],int]) -> (
        bytes, tuple[bytes, bytes]):
    """
    This method performs the merging of the most frequent token pair. In addition, it updates the token_pair_counts accordingly.
    @param token_pair_counts: The dictionary mapping from token pairs (represented as tuple[bytes]) to their count.
    @param pre_token_counts: The dictionary mapping from pre-tokens to their count.
    @return new_token: The newly generated token.
    @return merge: Token pair that was merged.
    """
    _, most_freq_pair = max([(count, token_pair) for token_pair, count in token_pair_counts.items()])
    new_token = most_freq_pair[0] + most_freq_pair[1]

    new_pre_token_counts: dict[tuple[bytes,...], int] = {}

    for pre_token, count in pre_token_counts.items():
        new_pre_token:list[bytes] = []
        i = 0
        while i < len(pre_token):
            if i < len(pre_token) - 1 and (pre_token[i], pre_token[i+1]) == most_freq_pair:
                new_pre_token.append(new_token)
                token_pair_counts[most_freq_pair] -= count
                if i > 0:
                    broken_token_pair = (pre_token[i - 1], pre_token[i])
                    token_pair_counts[broken_token_pair] -= count

                    new_token_pair = (pre_token[i-1], new_token)
                    token_pair_counts[new_token_pair] += count

                if i < len(pre_token) - 2:
                    broken_token_pair = (pre_token[i+1], pre_token[i+2])
                    token_pair_counts[broken_token_pair] -= count

                    new_token_pair = (new_token, pre_token[i+2])
                    token_pair_counts[new_token_pair] += count
                i+=2
            else:
                new_pre_token.append(pre_token[i])
                i += 1
        new_pre_token_counts[tuple(new_pre_token)] = count

    return new_token, most_freq_pair, new_pre_token_counts

if __name__ == "__main__":
    train_bpe("/Users/philip/PycharmProjects/Stanford-CS336-Assignment-1/tests/fixtures/tinystories_sample_5M.txt", 10000, ["<|endoftext|>"])
