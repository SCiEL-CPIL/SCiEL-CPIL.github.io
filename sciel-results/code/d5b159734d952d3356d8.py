"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def _within_radius(word, codeword, radius):
    return sum(x != y for x, y in zip(word, codeword)) <= radius


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Generate all words in the space
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)

    # Generate candidate codewords using direct sum of tetracode
    # For general q, n, radius: use a base code and extend
    if q == 3 and n >= 4:
        # Tetracode-based direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        candidates = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: use all words as candidates (expensive but correct)
        candidates = list(all_words)

    # Greedy set-cover: iteratively pick the candidate that covers the most uncovered words
    # Represent coverage as a boolean array
    covered = np.zeros(total_words, dtype=bool)
    selected = []
    
    # Precompute coverage sets for efficiency
    # For each candidate, compute which words it covers
    # This is expensive, so we do it lazily
    
    # Better approach: for each word, precompute which candidates cover it
    # Then greedily pick the candidate with max uncovered coverage
    
    # Actually, let's do a simpler greedy: at each step, for each uncovered word,
    # find the best candidate to cover it (one that also covers other uncovered words)
    
    # For efficiency with n=7, q=3: total_words=2187, candidates=243
    # We can afford O(total_words * candidates) per step
    
    # Precompute: for each candidate, the set of words it covers
    # Use a list of sets or bitmasks
    # 2187 bits fits in a Python int or a numpy array
    
    # Let's use numpy boolean arrays for coverage
    # coverage[c] = boolean array of which words candidate c covers
    
    # Precompute all coverage arrays
    coverage = []
    for cand in candidates:
        cov = np.array([sum(cand[i] != w[i] for i in range(n)) <= radius for w in all_words], dtype=bool)
        coverage.append(cov)
    
    # Greedy set cover
    remaining = np.ones(total_words, dtype=bool)
    remaining_count = total_words
    
    while remaining_count > 0:
        if time.time() > deadline - 5.0:
            break
        
        # Find the candidate that covers the most remaining words
        best_idx = -1
        best_count = -1
        for i, cov in enumerate(coverage):
            cnt = np.count_nonzero(remaining & cov)
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            break
        
        selected.append(candidates[best_idx])
        remaining &= ~coverage[best_idx]
        remaining_count = np.count_nonzero(remaining)
    
    # If we didn't finish in time, try to add a few more candidates to ensure coverage
    if remaining_count > 0:
        # Add candidates that cover the most remaining
        while remaining_count > 0 and len(selected) < len(candidates):
            best_idx = -1
            best_count = -1
            for i, cov in enumerate(coverage):
                if i in [selected.index(c) for c in [candidates[j] for j in range(len(candidates)) if candidates[j] == c]]:
                    continue
                cnt = np.count_nonzero(remaining & cov)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_idx == -1 or best_count == 0:
                break
            selected.append(candidates[best_idx])
            remaining &= ~coverage[best_idx]
            remaining_count = np.count_nonzero(remaining)
    
    return selected
# EVOLVE-BLOCK-END
