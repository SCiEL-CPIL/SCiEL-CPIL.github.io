# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, product

def _tetracode(q=3):
    """Ternary tetracode: 9 words of length 4, perfect radius-1 code."""
    return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]

def _direct_sum(codes_list):
    result = [list(cw) for cw in codes_list[0]]
    for code in codes_list[1:]:
        new_result = []
        for existing in result:
            for cw in code:
                new_result.append(existing + list(cw))
        result = new_result
    return result

def _is_covering(codewords, q, n, radius):
    """Check if codewords cover all words within radius."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for cw in codewords:
        for r in range(radius + 1):
            for pos in combinations(range(n), r):
                if r == 0:
                    idx = 0
                    for i, v in enumerate(cw):
                        idx = idx * q + v
                    covered[idx] = True
                else:
                    for vals in product(range(1, q), repeat=r):
                        w = list(cw)
                        for p, v in zip(pos, vals):
                            w[p] = (w[p] + v) % q
                        idx = 0
                        for i, v in enumerate(w):
                            idx = idx * q + v
                        covered[idx] = True
    return covered.all()

def _local_search(codewords, q, n, radius, time_limit, rng):
    """Try to reduce codeword count."""
    total = q ** n
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx
    
    # Precompute coverage bitsets
    coverage = []
    for cw in codewords:
        covered = np.zeros(total, dtype=bool)
        for r in range(radius + 1):
            for pos in combinations(range(n), r):
                if r == 0:
                    idx = word_to_idx(cw)
                    covered[idx] = True
                else:
                    for vals in product(range(1, q), repeat=r):
                        w = list(cw)
                        for p, v in zip(pos, vals):
                            w[p] = (w[p] + v) % q
                        covered[word_to_idx(w)] = True
        coverage.append(covered)
    
    current = list(codewords)
    current_cov = list(coverage)
    improved = True
    
    while improved and time.time() < time_limit:
        improved = False
        
        # Try removing each codeword if redundant
        for ci in range(len(current)):
            if time.time() >= time_limit:
                break
            new_covered = np.zeros(total, dtype=bool)
            for cj in range(len(current)):
                if cj != ci:
                    new_covered |= current_cov[cj]
            if new_covered.all():
                current.pop(ci)
                current_cov.pop(ci)
                improved = True
                break
    
    return current

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    rng = random.Random(42)
    
    # Strategy: direct sum of tetracodes where possible, then local search
    if q == 3 and n >= 4 and radius == 1:
        # Split n into groups of 4
        codes = []
        remaining = n
        while remaining >= 4:
            codes.append(_tetracode(q))
            remaining -= 4
        if remaining > 0:
            # For leftover coordinates, use all possible words
            leftover_code = [list(w) for w in np.ndindex(*([q] * remaining))]
            codes.append(leftover_code)
        
        codewords = _direct_sum(codes)
        
        # Local search to improve
        if time.time() < deadline - 5:
            codewords = _local_search(codewords, q, n, radius, deadline - 2, rng)
        
        return codewords
    
    # Fallback: greedy
    all_words = [list(w) for w in np.ndindex(*([q] * n))]
    return all_words  # Trivial covering (all words)
# EVOLVE-BLOCK-END
