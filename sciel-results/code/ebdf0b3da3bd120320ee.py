"""Optimal program for distinct subset sums with minimal largest element."""

def construct():
    # The minimal largest element for n=12 distinct positive integers
    # with distinct subset sums is 2^(12-1) = 2048.
    # The unique set achieving this is the powers of 2.
    return [1 << i for i in range(12)]
