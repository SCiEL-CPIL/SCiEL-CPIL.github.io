import time
import numpy as np

N = 18
MIN_DISTANCE = 6
TOTAL = 1 << N
DEADLINE_MARGIN = 5.0


def _seed_words():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    32 words, pairwise distance 8 (>= 6)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def _get_neighbors(center, radius):
    """Generate all words within Hamming distance <= radius of center."""
    results = [center]
    from itertools import combinations
    bits = list(range(N))
    for r in range(1, radius + 1):
        for comb in combinations(bits, r):
            w = center
            for b in comb:
                w ^= (1 << b)
            results.append(w)
    return results


def _build_forbidden_mask(words):
    """Build a boolean mask of all words within distance < MIN_DISTANCE of any codeword."""
    forbidden = np.zeros(TOTAL, dtype=bool)
    for w in words:
        neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
        forbidden[neighbors] = True
    return forbidden


def _greedy_vectorized(deadline, seed=12345):
    """
    Greedy construction using a precomputed forbidden mask.
    Uses a single vectorized pass to find all non-forbidden candidates,
    then iteratively adds them while updating the mask.
    """
    words = _seed_words()
    forbidden = _build_forbidden_mask(words)

    rng = np.random.default_rng(seed)

    # Precompute all candidates in random order for this seed
    candidates = np.arange(TOTAL, dtype=np.int64)
    rng.shuffle(candidates)

    # We will process candidates in chunks to allow deadline checks
    # and to avoid holding too many "accepted" words at once if the code grows large.
    # However, since we want the *largest* code, we should accept every valid candidate.
    
    # Optimization: 
    # 1. Find all candidates not in forbidden.
    # 2. For each, check if it conflicts with *other* newly accepted candidates? 
    #    No, the greedy algorithm adds one by one. 
    #    If we add a word, it forbids its neighbors. 
    #    If we have a batch of candidates, we must ensure they don't conflict with each other.
    
    # Strategy:
    # Iterate through candidates.
    # If candidate is not forbidden, add it, update forbidden mask.
    # To speed up "update forbidden mask", we precompute the neighbor indices for each possible word?
    # That would be 2^18 * 12616 integers, which is ~330MB. Too much memory.
    
    # Alternative: Update mask on the fly.
    # When we accept a word `w`, we need to mark `w` and its neighbors as forbidden.
    # We can precompute the list of neighbors for each word *on demand* and cache them?
    # Or just compute them. 12616 operations per accepted word.
    # If we accept ~100 words, that's 1.2M operations. Fast.
    # If we accept ~1000 words, that's 12M operations. Still fast in Python? Maybe slow.
    
    # Let's try to optimize the mask update.
    # Instead of a list of neighbors, can we use bit operations?
    # No, simple boolean array indexing is best.
    
    # To make it faster, let's precompute the neighbor indices for all 2^18 words?
    # Memory: 2^18 * 12616 * 8 bytes (int64) = 330 MB. 
    # This is risky but might be acceptable if memory allows. 
    # However, generating this takes time.
    
    # Let's stick to computing neighbors on the fly but optimize the inner loop.
    # Actually, the current program's bottleneck is likely the Python-level loop over candidates 
    # and the set operations.
    # Using a NumPy boolean array for `forbidden` makes the check `forbidden[c]` O(1) and very fast.
    # The update `forbidden[neighbors] = True` is also vectorized.
    
    # The main cost is generating `neighbors`.
    # Let's precompute a lookup table for neighbors?
    # No, let's just try the vectorized check first.
    
    accepted_count = 0
    start_time = time.time()
    
    # We can process all candidates in one go if we can batch the updates.
    # But updates depend on previous updates.
    # So we must iterate.
    
    # To speed up the iteration, we can use a "batch" approach:
    # 1. Find all candidates that are currently not forbidden.
    # 2. From these, select a subset that are mutually non-conflicting (independent set in the conflict graph).
    #    This is hard.
    # 3. Just iterate, but skip quickly.
    
    # Let's iterate, but use a NumPy array for the candidates to potentially vectorize the check?
    # No, the state changes.
    
    # Let's just loop, but the check is `if not forbidden[c]:` which is fast.
    # The update is the slow part.
    
    # Precompute neighbors for the seed words is already done.
    
    for c in candidates:
        if time.time() > deadline:
            break
        
        if not forbidden[c]:
            words.append(int(c))
            accepted_count += 1
            
            # Update forbidden mask
            neighbors = _get_neighbors(int(c), MIN_DISTANCE - 1)
            forbidden[neighbors] = True
            
            # Periodic check to avoid long pauses
            if accepted_count % 10 == 0:
                if time.time() > deadline:
                    break

    return words


def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN

    best_words = None
    best_len = 0

    # Strategy 1: Vectorized greedy with random order
    seeds = [42, 123, 999, 7777, 31415, 27182, 16180, 8675309, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        try:
            w = _greedy_vectorized(deadline, seed=seed)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass

    # Fallback
    if best_words is None:
        best_words = _seed_words()

    # Ensure distinctness and range
    seen = set()
    out = []
    for w in best_words:
        if w not in seen and 0 <= w < TOTAL:
            seen.add(w)
            out.append(w)
    if not out:
        out = _seed_words()

    return out
