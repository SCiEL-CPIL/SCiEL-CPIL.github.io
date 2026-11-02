"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _tetracode_words(q, n):
    """Return the direct-sum construction: tetracode on first 4 coords, all tails on rest."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    return [h + list(tail) for h in tetracode for tail in tails]


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _is_covering(codewords, q, n, radius):
    """Check if codewords cover all words within Hamming distance radius."""
    # Build a set of all covered words for efficiency
    covered = set()
    for cw in codewords:
        cw_tuple = tuple(cw)
        covered.add(cw_tuple)
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val != cw[pos]:
                        lst = list(cw)
                        lst[pos] = val
                        covered.add(tuple(lst))
    total = q ** n
    return len(covered) >= total


def _greedy_covering(codewords, q, n, radius):
    """Greedy: add words to cover uncovered space as much as possible."""
    covered = set()
    for cw in codewords:
        cw_tuple = tuple(cw)
        covered.add(cw_tuple)
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val != cw[pos]:
                        lst = list(cw)
                        lst[pos] = val
                        covered.add(tuple(lst))
    return len(covered)


def _random_word(q, n):
    return [random.randint(0, q - 1) for _ in range(n)]


def _local_search(codewords, q, n, radius, deadline):
    """Try to reduce code size via removal and replacement."""
    codewords = list(codewords)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing each codeword
        for i in range(len(codewords)):
            if time.time() > deadline - 5:
                break
            candidate = codewords[:i] + codewords[i+1:]
            if _is_covering(candidate, q, n, radius):
                codewords = candidate
                improved = True
                break
        # Try replacing a codeword with a random one
        if not improved and len(codewords) > 1:
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, len(codewords) - 1)
                new_word = _random_word(q, n)
                if new_word in codewords:
                    continue
                candidate = codewords[:i] + [new_word] + codewords[i+1:]
                if _is_covering(candidate, q, n, radius):
                    codewords = candidate
                    improved = True
                    break
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Start with the direct-sum construction
    codewords = _tetracode_words(q, n)
    
    # If n == 4, this is already optimal (9 words)
    if n == 4:
        return codewords

    # For larger n, try local search to reduce size
    if time.time() < deadline - 10:
        codewords = _local_search(codewords, q, n, radius, deadline)

    # Verify covering
    if not _is_covering(codewords, q, n, radius):
        # Fallback: regenerate
        codewords = _tetracode_words(q, n)

    return codewords
# EVOLVE-BLOCK-END
