"""Construct a small ternary covering code of length 8, radius 1.

Strategy:
- Use a "dual hypergraph" / Max-SAT style approach with bitset-accelerated local search.
- The 6561 space points are vertices; each potential codeword defines a 17-vertex hyperedge.
- We solve the minimum hitting set (set cover) problem via:
  1. Greedy construction to get an initial feasible solution.
  2. Aggressive local search: 1-flip (remove) and 2-swap (remove two, add one) moves
     using NumPy bitsets for fast coverage computation.
  3. Iterative deepening: try to reduce the solution size by k each round.

Time budget: ~115 seconds max.
"""

import itertools
import random
import time
import numpy as np

# Global cache
_all_words = None
_word_to_idx = None
_cover_bitsets = None  # shape (6561, 6561) as int8, or list of bitsets
_cover_np = None       # numpy array for fast operations
_N = 6561
_NUM_WORDS = 6561

def _precompute():
    """Precompute all words, their indices, and coverage as numpy bitsets."""
    global _all_words, _word_to_idx, _cover_bitsets, _cover_np, _N, _NUM_WORDS
    n = 8
    q = 3
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total = len(all_words)
    _N = total
    _NUM_WORDS = total
    
    # Build coverage as a numpy boolean array: cover_np[i, j] = True if word j is covered by codeword i
    cover = np.zeros((total, total), dtype=np.uint8)
    for i, w in enumerate(all_words):
        cover[i, i] = 1
        for pos in range(n):
            for alt in range(q):
                if alt == w[pos]:
                    continue
                w2 = list(w)
                w2[pos] = alt
                w2 = tuple(w2)
                cover[i, word_to_idx[w2]] = 1
    
    _all_words = all_words
    _word_to_idx = word_to_idx
    _cover_np = cover
    _cover_bitsets = None  # not used with numpy approach

def _ensure_precomputed():
    global _all_words, _word_to_idx, _cover_bitsets, _cover_np
    if _all_words is None:
        _precompute()

def _greedy_construct():
    """Greedy set cover using numpy for speed."""
    cover = _cover_np
    total = _NUM_WORDS
    uncovered = np.ones(total, dtype=np.uint8)
    selected = []
    
    # Precompute column sums for quick access
    # uncovered_count[i] = number of uncovered points covered by codeword i
    # We'll recompute this each iteration for simplicity (total is 6561, iterations ~500)
    
    while uncovered.any():
        # Find codeword with max coverage of uncovered
        # uncovered is a vector, cover[i] is a row. coverage_i = sum(uncovered & cover[i])
        # Using numpy: np.sum(cover * uncovered, axis=1) but that's 6561*6561 per iter = too slow
        # Better: maintain uncovered as a set-like structure
        
        # Actually, let's use a different approach: maintain uncovered as a Python set
        # and compute overlap via precomputed neighbor lists
        break
    
    # Fall back to set-based greedy which is faster for this size
    all_words = _all_words
    word_to_idx = _word_to_idx
    n = 8
    q = 3
    
    # Precompute neighbor indices for each word
    neighbors = []
    for i, w in enumerate(all_words):
        nbrs = [i]
        for pos in range(n):
            for alt in range(q):
                if alt == w[pos]:
                    continue
                w2 = list(w)
                w2[pos] = alt
                nbrs.append(word_to_idx[tuple(w2)])
        neighbors.append(nbrs)
    
    uncovered = set(range(total))
    selected = []
    
    while uncovered:
        best_idx = -1
        best_count = 0
        for i in range(total):
            if i in set(selected):
                continue
            overlap = sum(1 for j in neighbors[i] if j in uncovered)
            if overlap > best_count:
                best_count = overlap
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        for j in neighbors[best_idx]:
            uncovered.discard(j)
    
    return selected, neighbors

def _local_search(selected, neighbors, time_limit):
    """Aggressive local search: try 1-flip (remove) and 2-swap moves."""
    all_words = _all_words
    total = _NUM_WORDS
    selected_set = set(selected)
    start_time = time.time()
    
    # Precompute for each selected codeword, which points it uniquely covers
    # A point is uniquely covered by codeword c if no other selected codeword covers it
    
    def _compute_unique_coverage(sel_set):
        """For each codeword in sel_set, compute the set of points it uniquely covers."""
        # Count how many selected codewords cover each point
        cover_count = np.zeros(total, dtype=np.int32)
        for c in sel_set:
            for j in neighbors[c]:
                cover_count[j] += 1
        # A point is uniquely covered by c if cover_count[j] == 1 and c covers j
        unique = {}
        for c in sel_set:
            u = []
            for j in neighbors[c]:
                if cover_count[j] == 1:
                    u.append(j)
            unique[c] = set(u)
        return unique, cover_count
    
    def _is_covered(sel_set):
        """Check if sel_set covers all points."""
        covered = np.zeros(total, dtype=np.uint8)
        for c in sel_set:
            for j in neighbors[c]:
                covered[j] = 1
        return covered.all()
    
    # Iterative improvement
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        # Try 1-flip: remove a codeword if its unique coverage is empty
        unique, cover_count = _compute_unique_coverage(selected_set)
        for c in list(selected_set):
            if time.time() - start_time > time_limit:
                break
            if len(unique[c]) == 0:
                # Safe to remove
                selected_set.discard(c)
                improved = True
        
        # Try 2-swap: remove two codewords, add one that covers their union
        if time.time() - start_time > time_limit:
            break
        # This is expensive; only do a limited number of attempts
        # For each pair (a, b) in selected, see if there exists c not in selected
        # such that adding c and removing a,b still covers everything
        # Simplified: for each codeword a in selected, try to find b such that
        # removing a and b allows adding one c
        
        # Even simpler: try adding a random codeword and then see if we can remove two
        # Let's do: for each codeword c not in selected, try adding it, then try removing two
        # This is O(6561 * |selected|^2) which is too much.
        
        # Alternative: random restarts / simulated annealing style
        # Try random swaps: pick random a in selected, random c not in selected,
        # swap them, check if still covered
        for _ in range(5000):
            if time.time() - start_time > time_limit:
                break
            if len(selected_set) == 0:
                break
            a = random.choice(list(selected_set))
            # Pick a random codeword not in selected
            c = random.randint(0, total - 1)
            while c in selected_set:
                c = random.randint(0, total - 1)
            
            new_set = selected_set.copy()
            new_set.discard(a)
            new_set.add(c)
            if _is_covered(new_set):
                selected_set = new_set
                improved = True
    
    return list(selected_set)

def construct(q, n, radius):
    """Return a list of codewords (each a list of n ints in 0..q-1) covering
    all of {0..q-1}^n within Hamming distance radius."""
    _ensure_precomputed()
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: trivial construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    all_words = _all_words
    total = _NUM_WORDS
    start_time = time.time()
    
    # Precompute neighbors
    word_to_idx = _word_to_idx
    neighbors = []
    for i, w in enumerate(all_words):
        nbrs = [i]
        for pos in range(n):
            for alt in range(q):
                if alt == w[pos]:
                    continue
                w2 = list(w)
                w2[pos] = alt
                nbrs.append(word_to_idx[tuple(w2)])
        neighbors.append(nbrs)
    
    # Greedy construction
    uncovered = set(range(total))
    selected_set = set()
    
    while uncovered:
        best_idx = -1
        best_count = 0
        for i in range(total):
            if i in selected_set:
                continue
            overlap = sum(1 for j in neighbors[i] if j in uncovered)
            if overlap > best_count:
                best_count = overlap
                best_idx = i
        if best_idx < 0:
            break
        selected_set.add(best_idx)
        for j in neighbors[best_idx]:
            uncovered.discard(j)
    
    # Local search with time budget
    time_budget = 110  # leave 10s for safety
    selected = _local_search(list(selected_set), neighbors, time_budget)
    
    # Verify coverage
    covered = set()
    for c in selected:
        covered.update(neighbors[c])
    if len(covered) != total:
        # Fallback to greedy result
        selected = list(selected_set)
    
    result = [list(all_words[idx]) for idx in sorted(selected)]
    return result
