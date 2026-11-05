"""
Football pool problem for 6 matches: ternary covering code K_3(6,1)

Strategy: Start from the known 73-word construction (van Laarhoven et al.) and
attempt to reduce the number of codewords via greedy local search / hill climbing.

The seed program uses a direct product construction giving 9 * 3^(n-4) words.
For n=6, that's 9 * 9 = 81 words.

The known best is 73 words. We'll try to find a 73-word code or better.

Approach:
1. Generate a good initial code (the 81-word direct product, or a smarter initial set)
2. Use local search: try removing codewords one at a time, checking if coverage is maintained
3. Use greedy set-cover style construction to build a smaller code from scratch
"""

import itertools
import time
import random
import sys

# Use numpy if available for faster operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _word_to_int(word, q, n):
    """Convert a word (list of ints) to an integer index in [0, q^n)."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _int_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _hamming_distance(w1, w2):
    """Compute Hamming distance between two words."""
    return sum(1 for a, b in zip(w1, w2) if a != b)


def _neighbors(word, q, n):
    """Generate all words within Hamming distance 1 of the given word."""
    # The word itself
    yield word
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                new_word = list(word)
                new_word[i] = v
                yield new_word


def _is_covered(word, codewords, q, n, radius):
    """Check if a word is within radius of any codeword."""
    for cw in codewords:
        if _hamming_distance(word, cw) <= radius:
            return True
    return False


def _coverage_check(codewords, q, n, radius):
    """Check if all words are covered. Returns True if fully covered."""
    total = q ** n
    covered = [False] * total
    
    for cw in codewords:
        cw_idx = _word_to_int(cw, q, n)
        for i in range(n):
            orig = cw[i]
            for v in range(q):
                if v != orig:
                    new_word = list(cw)
                    new_word[i] = v
                    covered[_word_to_int(new_word, q, n)] = True
        covered[cw_idx] = True
    
    return all(covered)


def _build_initial_code(q, n, radius):
    """Build an initial code using direct product construction."""
    # For ternary codes, use the tetracode (perfect code of length 4, radius 1)
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        rest = n - 4
        code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=rest):
                code.append(h + list(tail))
        return code
    else:
        # Fallback: greedy construction
        return _greedy_construct(q, n, radius)


def _greedy_construct(q, n, radius):
    """Greedy set-cover construction."""
    total = q ** n
    covered = [False] * total
    code = []
    
    # Precompute all words and their neighborhoods
    all_words = [_int_to_word(i, q, n) for i in range(total)]
    
    # For each word, compute which words it covers
    # We'll use a priority: pick the word that covers the most uncovered words
    
    uncovered_count = total
    while uncovered_count > 0:
        best_word = None
        best_gain = -1
        
        for i in range(total):
            if covered[i]:
                continue
            w = all_words[i]
            gain = 0
            for v in _neighbors(w, q, n):
                v_idx = _word_to_int(v, q, n)
                if not covered[v_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = w
        
        if best_word is None:
            break
        
        code.append(best_word)
        for v in _neighbors(best_word, q, n):
            v_idx = _word_to_int(v, q, n)
            if not covered[v_idx]:
                covered[v_idx] = True
                uncovered_count -= 1
    
    return code


def _try_remove_codeword(code, idx, q, n, radius):
    """Try to remove codeword at index idx. Return new code if coverage maintained."""
    new_code = code[:idx] + code[idx+1:]
    if _coverage_check(new_code, q, n, radius):
        return new_code
    return None


def _local_search_reduce(code, q, n, radius, max_iter=5000, timeout=None):
    """Local search: repeatedly try to remove codewords."""
    start_time = time.time()
    code = list(code)
    
    for iteration in range(max_iter):
        if timeout is not None and (time.time() - start_time) > timeout:
            break
        
        improved = False
        # Try removing each codeword
        for i in range(len(code)):
            new_code = _try_remove_codeword(code, i, q, n, radius)
            if new_code is not None:
                code = new_code
                improved = True
                break
        
        if not improved:
            break
    
    return code


def _swap_search(code, q, n, radius, max_iter=10000, timeout=None):
    """Try swapping one codeword with another to maintain coverage with same size,
    or find a smaller code."""
    start_time = time.time()
    code = list(code)
    total = q ** n
    all_words = [_int_to_word(i, q, n) for i in range(total)]
    
    for iteration in range(max_iter):
        if timeout is not None and (time.time() - start_time) > timeout:
            break
        
        improved = False
        # Try removing each codeword
        for i in range(len(code)):
            new_code = _try_remove_codeword(code, i, q, n, radius)
            if new_code is not None:
                code = new_code
                improved = True
                break
        
        if not improved:
            break
    
    return code


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet of size q with covering radius radius."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Build initial code
    code = _build_initial_code(q, n, radius)
    
    # Verify initial code is valid
    if not _coverage_check(code, q, n, radius):
        # Fallback to greedy
        code = _greedy_construct(q, n, radius)
    
    # Try to reduce the code size
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5:
        code = _local_search_reduce(code, q, n, radius, max_iter=10000, timeout=remaining_time - 5)
    
    # Final verification
    if not _coverage_check(code, q, n, radius):
        # Rebuild from scratch if something went wrong
        code = _build_initial_code(q, n, radius)
    
    return code
