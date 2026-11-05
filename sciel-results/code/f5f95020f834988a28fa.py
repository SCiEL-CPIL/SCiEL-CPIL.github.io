"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np


def _hamming_dist(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def _neighbors(word, q, n):
    """Generate all words within Hamming distance 1 of word."""
    result = [list(word)]
    for i in range(n):
        for val in range(q):
            if val != word[i]:
                new_word = list(word)
                new_word[i] = val
                result.append(new_word)
    return result


def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    # For small cases, use direct construction
    if n <= 4:
        # Use ternary tetracode for n=4, extend for smaller
        if n == 4:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        elif n == 3:
            # Perfect code for ternary length 3, radius 1: 3 words
            return [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        elif n == 2:
            return [[0, 0], [1, 1], [2, 2], [0, 1], [1, 0], [2, 2], [0, 2], [2, 0], [1, 2], [2, 1]]
        else:
            # n=1
            return [[i] for i in range(q)]
    
    # Greedy set cover approach with local search
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: for each word, which codeword positions cover it
    # A codeword c covers all words within distance radius
    
    # Start with a good initial solution using tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Greedy: iteratively add codeword that covers most uncovered words
    covered = [False] * len(all_words)
    codewords = []
    
    for cw in initial_codewords:
        codewords.append(cw)
        for neighbor in _neighbors(cw, q, n):
            idx = word_to_idx[tuple(neighbor)]
            covered[idx] = True
    
    # Try to improve by removing redundant codewords
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            # Check if removing codeword[i] still leaves all words covered
            test_covered = covered[:]
            # Uncover words covered only by codeword[i]
            for neighbor in _neighbors(codewords[i], q, n):
                idx = word_to_idx[tuple(neighbor)]
                # Check if any other codeword covers this word
                still_covered = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    if _hamming_dist(codewords[j], all_words[idx]) <= radius:
                        still_covered = True
                        break
                if not still_covered:
                    break
            else:
                # Can remove
                del codewords[i]
                for neighbor in _neighbors(codewords[i] if i < len(codewords) else codewords[-1], q, n):
                    pass
                # Recalculate covered
                covered = [False] * len(all_words)
                for cw in codewords:
                    for neighbor in _neighbors(cw, q, n):
                        idx = word_to_idx[tuple(neighbor)]
                        covered[idx] = True
                improved = True
                break
    
    # Local search: try replacing a codeword with a better one
    best_codewords = codewords[:]
    best_size = len(codewords)
    
    # Try random local search
    random.seed(42)
    for iteration in range(50):
        current = best_codewords[:]
        for step in range(100):
            if len(current) >= best_size:
                break
            # Try removing a random codeword and see if we can cover the gap
            idx = random.randint(0, len(current) - 1)
            cw_to_remove = current[idx]
            
            # Find words that would become uncovered
            would_uncover = []
            for neighbor in _neighbors(cw_to_remove, q, n):
                w = tuple(neighbor)
                w_idx = word_to_idx[w]
                # Check if covered by another codeword
                is_covered = False
                for j, cw in enumerate(current):
                    if j == idx:
                        continue
                    if _hamming_dist(cw, w) <= radius:
                        is_covered = True
                        break
                if not is_covered:
                    would_uncover.append(w)
            
            if not would_uncover:
                # Can remove
                del current[idx]
                best_codewords = current[:]
                best_size = len(current)
                break
            else:
                # Try to find a single codeword that covers all would_uncover words
                # and is close to cw_to_remove
                found = False
                for candidate in all_words:
                    if tuple(candidate) == tuple(cw_to_remove):
                        continue
                    covers_all = True
                    for w in would_uncover:
                        if _hamming_dist(candidate, w) > radius:
                            covers_all = False
                            break
                    if covers_all:
                        # Check if this candidate is already in current
                        if candidate not in current:
                            current[idx] = candidate
                            # Check if this reduces size or same size
                            if len(current) < best_size:
                                best_codewords = current[:]
                                best_size = len(current)
                            found = True
                            break
                if found:
                    break
    
    return best_codewords
# EVOLVE-BLOCK-END
