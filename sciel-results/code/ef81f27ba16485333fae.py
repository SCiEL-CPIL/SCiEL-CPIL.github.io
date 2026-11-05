"""Improved construction for ternary covering code K_3(8,1).

Strategy:
1. Start from the direct-sum of the ternary tetracode (9 * 3^4 = 729 words).
2. Use randomized greedy peeling: repeatedly remove a codeword if its covered
   ball is fully covered by the remaining codewords.
3. To improve on the seed's fixed-order peeling, use multiple random restarts
   with different random orderings, and keep the best (smallest) result.
4. Add a deadline guard to ensure we finish well within the time limit.
"""

import itertools
import time
import random
import numpy as np


def _ball_mask(word, q, n):
    """Return a boolean array of length q^n marking words within Hamming distance 1 of `word`."""
    size = q ** n
    mask = np.zeros(size, dtype=bool)
    # The word itself
    idx = 0
    base = 1
    for i in range(n):
        idx += word[i] * base
        base *= q
    mask[idx] = True
    # Words at distance 1
    base = 1
    for i in range(n):
        idx_i = idx - word[i] * base
        for v in range(q):
            if v != word[i]:
                mask[idx_i + v * base] = True
        base *= q
    return mask


def _build_codebook(q, n):
    """Precompute ball masks for all q^n words and the tetracode-based initial code."""
    size = q ** n
    # Precompute all ball masks
    all_balls = np.zeros((size, size), dtype=bool)
    for w in range(size):
        # Convert integer to word
        word = [0] * n
        tmp = w
        for i in range(n):
            word[i] = tmp % q
            tmp //= q
        all_balls[w] = _ball_mask(word, q, n)
    # Initial code: tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial.append(h + list(tail))
    return all_balls, initial


def _peel(all_balls, initial, q, n, seed=None, time_budget=None, start_time=None):
    """Greedy peeling: remove codewords whose balls are fully covered by others.
    Returns the list of remaining codeword indices (into the full space)."""
    size = q ** n
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Convert initial words to integer indices
    def word_to_idx(word):
        idx = 0
        base = 1
        for v in word:
            idx += v * base
            base *= q
        return idx
    
    code_indices = [word_to_idx(w) for w in initial]
    in_code = np.zeros(size, dtype=bool)
    for ci in code_indices:
        in_code[ci] = True
    
    # Coverage count: how many codewords cover each word
    coverage = np.zeros(size, dtype=np.int32)
    for ci in code_indices:
        coverage += all_balls[ci]
    
    # Random order for peeling
    order = list(range(len(code_indices)))
    rng.shuffle(order)
    
    # Track which codewords are still present
    present = [True] * len(code_indices)
    
    # For each codeword, count how many words it uniquely covers
    # A codeword can be removed if for every word in its ball, coverage > 1
    # i.e., min(coverage over ball) >= 2
    
    # Precompute: for each codeword, the minimum coverage in its ball
    # We'll update this incrementally
    
    # To speed up: only consider codewords in random order
    removed_count = 0
    for pos in order:
        if time_budget is not None and start_time is not None:
            if time.time() - start_time > time_budget:
                break
        if not present[pos]:
            continue
        ci = code_indices[pos]
        # Check if all words in this codeword's ball have coverage >= 2
        ball = all_balls[ci]
        # Find min coverage in ball
        # We need: for all j where ball[j], coverage[j] >= 2
        # Equivalently: no j in ball has coverage[j] == 1
        # i.e., sum of (coverage == 1) over ball == 0
        # But computing this naively is O(q^n) per check.
        # Instead, track unique coverage count.
        
        # Let's use a different approach: count how many words in the ball have coverage == 1
        # We can precompute a "unique" flag and maintain counts.
        
        # For efficiency, let's just check: is there any word in the ball with coverage == 1?
        # We'll use numpy: 
        # unique_in_ball = np.sum((coverage == 1) & ball)
        # If 0, we can remove.
        
        # But this is O(size) per check, and we have up to 729 checks, each O(6561) = ~4.8M ops.
        # That's fine for 729 codewords.
        
        # Actually, let's be smarter. Let's maintain for each codeword a count of
        # words it uniquely covers. When we remove a codeword, we update these counts.
        # This is more complex but faster for many removals.
        
        # For simplicity and given the small number of codewords (729), let's do the direct check.
        unique_count = np.sum((coverage == 1) & ball)
        if unique_count == 0:
            # Remove this codeword
            present[pos] = False
            removed_count += 1
            # Update coverage
            coverage -= all_balls[ci]
    
    # Collect remaining codewords
    remaining = [code_indices[pos] for pos in range(len(code_indices)) if present[pos]]
    return remaining


def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Build codebook
    all_balls, initial = _build_codebook(q, n)
    
    # Convert initial words to indices for the result
    def word_to_idx(word):
        idx = 0
        base = 1
        for v in word:
            idx += v * base
            base *= q
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
    
    # Initial code as a baseline
    best = initial[:]
    best_len = len(initial)
    
    # Try multiple random restarts of peeling
    elapsed = time.time() - start_time
    max_time_for_peeling = deadline - elapsed - 5.0  # 5s buffer for finalization
    
    if max_time_for_peeling > 1.0:
        # Estimate how many restarts we can afford
        # Each peeling run takes roughly: 729 * 6561 / 1e8 seconds ≈ 0.05s (optimistic)
        # Let's just try as many as we can
        n_restarts = int(max_time_for_peeling / 0.05)
        n_restarts = max(1, min(n_restarts, 50))
        
        for restart in range(n_restarts):
            if time.time() - start_time > deadline - 5.0:
                break
            seed = restart * 1000 + 42
            remaining = _peel(all_balls, initial, q, n, seed=seed,
                              time_budget=deadline - (time.time() - start_time),
                              start_time=start_time)
            if len(remaining) < best_len:
                best = [idx_to_word(idx) for idx in remaining]
                best_len = len(remaining)
    
    return best


# Precompute at module level for speed? No, let's keep it in construct.
# But we can cache the all_balls computation if called multiple times.
# For now, this should be fine.
