"""Football pool problem for 7 matches: ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

# Precompute the tetracode (perfect code of length 4 over GF(3), radius 1)
_TETRACODE = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)], dtype=np.int8)
_TETRA_SIZE = 9
_TETRA_TAILS = np.array(list(product(range(3), repeat=3)), dtype=np.int8)  # 27 tails of length 3
_TETRA_TAILS_FLAT = _TETRA_TAILS.reshape(-1)  # 27


def _direct_sum_initial(n):
    """Direct sum: 9 * 3**(n-4) words."""
    if n == 4:
        return _TETRACODE.tolist()
    tails = np.array(list(product(range(3), repeat=n - 4)), dtype=np.int8)
    words = np.empty((len(_TETRACODE) * len(tails), n), dtype=np.int8)
    for i, t in enumerate(_TETRACODE):
        words[i * len(tails):(i + 1) * len(tails), :4] = t
        words[i * len(tails):(i + 1) * len(tails), 4:] = tails
    return words


def _is_covering(words, q, n, radius):
    """Check if every word in {0..q-1}^n is within radius of some codeword."""
    # For n=7, q=3, radius=1: total 2187 words
    # Each codeword covers 1 + n*(q-1) = 1 + 7*2 = 15 words
    # Use a boolean array for covered words
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    
    # Convert words to flat indices
    words = np.asarray(words, dtype=np.int8)
    
    for w in words:
        # Mark w itself
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        covered[idx] = True
        
        # Mark all words at distance 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                idx = 0
                for i in range(n):
                    v = val if i == pos else w[i]
                    idx = idx * q + v
                covered[idx] = True
    
    return covered.all()


def _greedy_cover(q, n, radius, time_limit, rng=None):
    """Greedy set cover with bitmask optimization."""
    if rng is None:
        rng = random.Random()
    
    total = q ** n
    # Precompute all words as flat indices
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int8)
    
    # For each codeword position, precompute which words it covers
    # A codeword at position p covers p and all words at distance <= radius
    # For radius=1, n=7, q=3: covers 15 words
    
    # Map from flat index to set of words covered
    # We'll use a numpy array: for each of the total possible positions, 
    # store the 15 flat indices it covers (or fewer if radius allows)
    
    # Actually, let's think differently. The universe is all_words (2187 elements).
    # Each candidate codeword is one of the 2187 positions.
    # We need to select minimum number of positions such that every word is covered.
    
    # Precompute coverage: for each candidate position, which words does it cover?
    # coverage[candidate_idx] = list of word_indices covered
    
    # For efficiency, precompute the coverage sets as bitmasks or sorted arrays
    # 2187 candidates, each covers 15 words -> manageable
    
    # Convert all_words to flat indices
    flat_indices = np.zeros(total, dtype=np.int32)
    for i, w in enumerate(all_words):
        idx = 0
        for j in range(n):
            idx = idx * q + w[j]
        flat_indices[i] = idx
    
    # Precompute coverage for each candidate
    # coverage[c] = set of flat_indices covered by codeword at position c
    # Use a list of sets for speed
    coverage = [set() for _ in range(total)]
    for c in range(total):
        w = all_words[c]
        # Cover w itself
        idx = 0
        for j in range(n):
            idx = idx * q + w[j]
        coverage[c].add(idx)
        # Cover distance-1 neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                idx = 0
                for j in range(n):
                    v = val if j == pos else w[j]
                    idx = idx * q + v
                coverage[c].add(idx)
    
    # Greedy: repeatedly pick the candidate that covers the most uncovered words
    uncovered = set(range(total))
    selected = []
    start_time = time.time()
    
    while uncovered:
        if time.time() - start_time > time_limit - 5:
            break
        
        best_c = -1
        best_count = 0
        # Randomize tie-breaking
        for c in range(total):
            cnt = len(coverage[c] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_c = c
            elif cnt == best_count and cnt > 0:
                if rng.random() < 0.1:
                    best_c = c
        
        if best_c == -1 or best_count == 0:
            break
        
        selected.append(best_c)
        uncovered -= coverage[best_c]
    
    if not uncovered:
        return [all_words[c].tolist() for c in selected]
    return None


def _local_search(words, q, n, radius, time_limit, rng=None):
    """Try to reduce the code size by removing redundant words or swapping."""
    if rng is None:
        rng = random.Random()
    
    words = [list(w) for w in words]
    start_time = time.time()
    
    # Try removing words one at a time
    improved = True
    while improved:
        improved = False
        for i in range(len(words)):
            if time.time() - start_time > time_limit - 5:
                break
            # Try removing word i
            candidate = words[:i] + words[i+1:]
            if _is_covering(candidate, q, n, radius):
                words = candidate
                improved = True
                break  # Restart loop since indices changed
    
    return words


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet size q with covering radius radius."""
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for the 120s limit
    
    rng = random.Random(42)
    
    # Start with direct sum construction
    words = _direct_sum_initial(n)
    best_words = [w.tolist() for w in words]
    best_size = len(best_words)
    
    # For n=7, q=3, radius=1: direct sum gives 9*27 = 243 words
    # Try to improve with greedy + local search
    
    if n == 7 and q == 3 and radius == 1:
        # Try greedy set cover
        greedy_result = _greedy_cover(q, n, radius, deadline - start_time - 10, rng)
        if greedy_result is not None:
            if len(greedy_result) < best_size:
                best_words = greedy_result
                best_size = len(best_words)
        
        # Try local search on the best so far
        if time.time() < deadline - 10:
            ls_result = _local_search(best_words, q, n, radius, deadline - start_time - 5, rng)
            if ls_result and len(ls_result) < best_size:
                best_words = ls_result
                best_size = len(best_words)
    
    return best_words
# EVOLVE-BLOCK-END
