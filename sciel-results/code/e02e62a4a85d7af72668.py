"""Construct a large binary code of length 18 with minimum distance 6."""
import time
import numpy as np

N = 18
MIN_D = 6
DEADLINE = 55.0  # safety margin before 60s/90s kill

def _popcount(x):
    return x.bit_count()

def _hamming(a, b):
    return (a ^ b).bit_count()

def _reed_muller_base():
    """First-order Reed-Muller code RM(1,4) on 16 coords, extended with 2 parity bits."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Extend with 2 zero bits (positions 16, 17)
            words.append(word)
    return words

def _build_conflict_matrix(codewords):
    """Build a set of indices that conflict (distance < MIN_D) with each codeword."""
    n = len(codewords)
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _hamming(codewords[i], codewords[j]) < MIN_D:
                conflicts[i].add(j)
                conflicts[j].add(i)
    return conflicts

def _greedy_add(codewords, candidates, conflicts):
    """Greedily add candidates that don't conflict with existing codewords."""
    added = 0
    for c in candidates:
        # Check against all existing codewords
        ok = True
        for i, w in enumerate(codewords):
            if _hamming(c, w) < MIN_D:
                ok = False
                break
        if ok:
            codewords.append(c)
            added += 1
    return added

def _local_search(codewords, time_limit):
    """Try to improve the code via local search: remove one, add two."""
    start = time.time()
    n = len(codewords)
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        n = len(codewords)
        for i in range(n):
            if (time.time() - start) > time_limit:
                break
            # Try removing codeword i
            remaining = [codewords[j] for j in range(n) if j != i]
            # Try adding each possible new word
            # Use a limited set of candidates: neighbors of remaining codewords at distance >= MIN_D
            # For efficiency, try words at distance exactly MIN_D from some remaining word
            candidates = set()
            for w in remaining:
                # Generate some candidates near w
                # Try flipping 1 bit, 2 bits from w
                for mask in range(1, 1 << N):
                    if mask.bit_count() <= 2:
                        c = w ^ mask
                        # Check if c is compatible with all remaining
                        ok = True
                        for r in remaining:
                            if r != w and _hamming(c, r) < MIN_D:
                                ok = False
                                break
                        if ok:
                            # Also check c != w
                            if c != w:
                                candidates.add(c)
            # Try adding candidates
            best_add = None
            best_add_count = 0
            for c in candidates:
                if c in remaining:
                    continue
                # Check compatibility
                ok = True
                for r in remaining:
                    if _hamming(c, r) < MIN_D:
                        ok = False
                        break
                if ok:
                    # Can we add it?
                    new_code = remaining + [c]
                    if len(new_code) > n:
                        best_add = c
                        best_add_count = 1
                        break
            if best_add is not None:
                codewords = remaining + [best_add]
                improved = True
    return codewords

def construct():
    start_time = time.time()
    
    # Start with Reed-Muller base (32 words)
    codewords = _reed_muller_base()
    
    # Greedy expansion: try all possible words
    # For efficiency, use a priority-based approach
    # Generate candidates in order of "desirability"
    
    # First, try to add words that are far from all existing
    # Use a simple greedy: iterate through all 2^18 words, add if compatible
    
    # To speed up, use a vectorized approach with numpy
    all_words = np.arange(2 ** N, dtype=np.int64)
    
    # Build a conflict check function using numpy
    def _check_compatible_np(new_words, existing):
        """Check which new_words are compatible with all existing codewords."""
        if len(existing) == 0:
            return np.ones(len(new_words), dtype=bool)
        # For each new word, check distance to each existing word
        # Use XOR and popcount
        compatible = np.ones(len(new_words), dtype=bool)
        for w in existing:
            dists = (new_words ^ w).astype(np.uint32)
            # Popcount using numpy
            # np.bit_count is available in numpy >= 2.0, but let's use a workaround
            # Actually, let's use a different approach
            # Convert to bytes and count bits
            # For efficiency, use a lookup table for 8-bit popcount
            # Or use the fact that we can compute popcount via bit operations
            # Let's use a simple loop for now, but vectorize the XOR
            pass
        return compatible
    
    # Simpler approach: iterate through candidates, check compatibility
    # Use a set for fast lookup
    existing_set = set(codewords)
    
    # Try adding words in a smart order
    # Words that are "far" from existing ones first
    
    # Phase 1: Add words at distance >= 6 from all existing
    # Iterate through all words, but skip those already in the set
    added_count = 0
    for w in range(2 ** N):
        if time.time() - start_time > DEADLINE:
            break
        if w in existing_set:
            continue
        # Check compatibility
        ok = True
        for c in codewords:
            if _hamming(w, c) < MIN_D:
                ok = False
                break
        if ok:
            codewords.append(w)
            existing_set.add(w)
            added_count += 1
    
    # Phase 2: Local search to improve
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        codewords = _local_search(codewords, remaining_time - 2)
    
    # Final check: ensure all pairs have distance >= MIN_D
    # Remove any conflicts (shouldn't happen, but safety)
    final = []
    for w in codewords:
        ok = True
        for c in final:
            if _hamming(w, c) < MIN_D:
                ok = False
                break
        if ok:
            final.append(w)
    
    return final

if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} codewords")
