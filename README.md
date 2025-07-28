# CS336 Spring 2025 Assignment 1: Basics

This repository contains my implementation for Assignment 1 of the "CS336: Language Modeling from Scratch" course. The assignment focuses on foundational principles of language modeling.

Consistent with the course's emphasis on building systems "from scratch," all code within this repository was developed entirely by me, without the use of any AI assistance.

For my own understanding, this README contains descriptions of all individual parts of the implementation.

# Part 1: Byte-Pair Encoding Tokenizer

## Motivation
For applying any type of machine learning algorithm to natural language, we need a numerical representation of our language. Tokenizers are systems that convert arbitrary sequences of words into sequences of integers. One intuitive idea for tokenizers is utilizing unicode encodings such as UTF-8. Per definition, UTF-8 represents arbitrary sequences of text as sequences of bytes. We can interpret this as a sequence of integer IDs where the vocabulary is the set of possible byte values [0,255]. While very elegant, a major drawback of this tokenizer is its compression ratio (number of bytes in original sequence / number of tokens produced by tokenizer) of 1. Improving this ratio by merging tokens gives way to the byte-pair encoding (bpe) tokenizer widely used in practice and implemented as part of this assignment.

## Implementation
The bpe tokenizer works by iteratively merging the two most common consecutive tokens into a new token which is added to the vocabulary. Note that it does not necessarily merge exactly two bytes into a new token as the name might indicate. The bpe training consists of three main phases:

### Vocabulary Initialization 
The starting vocabulary for our bpe tokenizer is the one-to-one mapping from bytestring token to integer ID. In Python this is represented by a dict: bytes -> int. Vocab = {i:i for i in range(256)}

### Pre-Tokenization
Instead of counting byte-pairs directly on our input sequence, we can pre-tokenize it and count within the pre-tokens. This doesn't only improve efficiency but also enforces some semantic rules on the new vocabulary entries generated.

### Compute BPE Merges
For computing the bpe merges, we need to find the most frequent token pair. Each occurrence of this pair is then replaced by the merged token (which is added to the vocabulary).
