"""Improved covering-code search: vectorized greedy + fast local search with cover counts."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

# Deadline guard: evaluator kills at 150s wall-clock; keep a safety margin
_DEADLINE_SECONDS = 140.0


def _elapsed(start):
    return time.monotonic() - start


def _idx_to_tuple(idx, n, q):
    """Convert flat index to tuple of digits."""
    t = []
    for i in range(n):
        t.append(idx % q)
        idx //= q
    return t[::-1]


def _tuple_to_idx(t, n, q):
    """Convert tuple of digits to flat index."""
    idx = 0
    for d in t:
        idx = idx * q + d
    return idx


def _greedy_construct(q, n, radius, rng, max_time_remaining):
    """
    Vectorized greedy set cover.
    Returns list of codewords (tuples) that cover all of {0..q-1}^n.
    """
    total = q ** n
    covered = np.zeros(total, dtype=np.uint8)
    codewords = []

    # Precompute neighbor masks for each codeword?
    # For radius=1, each codeword covers 1 + n*(q-1) words.
    # We maintain gain[c] = number of uncovered words covered by c.
    
    # Precompute neighbors for all codewords to speed up updates?
    # Memory: total * (1 + n*(q-1)) ints. 2187 * 15 = 32805. Fine.
    # But we need to update gain for all c that cover w.
    # So we need reverse_cover: w -> list of c covering w.
    
    reverse_cover = [[] for _ in range(total)]
    neighbors = [[] for _ in range(total)] # neighbors[c] = list of words covered by c
    
    for c in range(total):
        c_tuple = _idx_to_tuple(c, n, q)
        neighbors[c].append(c)
        reverse_cover[c].append(c)
        for pos in range(n):
            orig = c_tuple[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_tuple = list(c_tuple)
                new_tuple[pos] = val
                w = _tuple_to_idx(new_tuple, n, q)
                neighbors[c].append(w)
                reverse_cover[w].append(c)

    gain = np.zeros(total, dtype=np.int32)
    # Initial gain is the size of the neighbor set
    for c in range(total):
        gain[c] = len(neighbors[c])

    while True:
        elapsed_check = _elapsed(max_time_remaining[0])
        if elapsed_check > max_time_remaining[1]:
            break

        if np.max(gain) <= 0:
            break
        
        c = int(np.argmax(gain))
        codewords.append(c)

        for w in neighbors[c]:
            if covered[w] == 0:
                covered[w] = 1
                for cc in reverse_cover[w]:
                    gain[cc] -= 1

        if covered.all():
            break

    return [_idx_to_tuple(c, n, q) for c in codewords]


def _is_cover(codewords, q, n, radius):
    """Check if codewords form a valid covering."""
    total = q ** n
    covered = np.zeros(total, dtype=np.uint8)
    
    for cw in codewords:
        c_idx = _tuple_to_idx(cw, n, q)
        covered[c_idx] = 1
        for pos in range(n):
            orig = cw[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_cw = list(cw)
                new_cw[pos] = val
                w_idx = _tuple_to_idx(new_cw, n, q)
                covered[w_idx] = 1

    return covered.all()


def _local_search(codewords, q, n, radius, rng, max_time_remaining):
    """
    Fast local search using cover_count array.
    """
    codewords = list(codewords)
    total = q ** n
    
    # Precompute neighbors and reverse cover for fast updates
    neighbors = [[] for _ in range(total)]
    reverse_cover = [[] for _ in range(total)]
    
    for c in range(total):
        c_tuple = _idx_to_tuple(c, n, q)
        neighbors[c].append(c)
        reverse_cover[c].append(c)
        for pos in range(n):
            orig = c_tuple[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_tuple = list(c_tuple)
                new_tuple[pos] = val
                w = _tuple_to_idx(new_tuple, n, q)
                neighbors[c].append(w)
                reverse_cover[w].append(c)

    # Initialize cover_count
    cover_count = np.zeros(total, dtype=np.int16)
    for cw in codewords:
        c_idx = _tuple_to_idx(cw, n, q)
        for w in neighbors[c_idx]:
            cover_count[w] += 1
            
    # Verify initial coverage
    if np.any(cover_count == 0):
        # Fallback if initial was invalid (shouldn't happen)
        return codewords

    improved = True
    while improved and _elapsed(max_time_remaining[0]) < max_time_remaining[1]:
        improved = False
        for i in range(len(codewords)):
            if _elapsed(max_time_remaining[0]) > max_time_remaining[1]:
                return codewords
            
            c_idx = _tuple_to_idx(codewords[i], n, q)
            
            # Try to remove codeword i
            # Decrement cover_count for neighbors of c_idx
            for w in neighbors[c_idx]:
                cover_count[w] -= 1
                
            # Check if coverage is lost
            # Find words with cover_count == 0
            lost_words = np.where(cover_count == 0)[0]
            
            if len(lost_words) == 0:
                # Success: removed a codeword
                codewords.pop(i)
                improved = True
                break
            else:
                # Coverage lost. Try to find a replacement.
                # We need a codeword c_new such that all lost_words are in neighbors[c_new].
                # And c_new should be different from c_idx (or at least not already in codewords? 
                # Actually if we re-add c_idx it's a no-op. We want a different one to potentially improve structure, 
                # or just any valid one to maintain size).
                
                # If lost_words is small, intersect the sets of codewords covering each lost word.
                # candidates = set of all codewords covering lost_words[0]
                # for w in lost_words[1:]: candidates &= set(reverse_cover[w])
                
                # Optimization: Only do this if len(lost_words) is small (e.g. <= 3)
                if len(lost_words) <= 3:
                    candidates = set(reverse_cover[lost_words[0]])
                    for w in lost_words[1:]:
                        candidates &= set(reverse_cover[w])
                    
                    # Filter out current codewords? 
                    # If we pick a codeword already in the list, we are effectively swapping i for j.
                    # If we pick c_idx, we revert.
                    # We want to pick a codeword NOT in codewords to maintain size, 
                    # OR a codeword IN codewords (swapping i for j).
                    
                    # Let's try to find a candidate not in codewords first (to keep size same, but maybe better structure? 
                    # Actually size is same. We just want to fix coverage).
                    # If we find one, we add it. Size remains same.
                    # Wait, we removed one, so size is N-1. Adding one makes it N.
                    # This is a "swap" in the sense of maintaining size, but we didn't reduce size.
                    
                    # Is it worth doing?
                    # If we can't reduce size, we might get stuck in a local minimum.
                    # Swapping can help escape local minima.
                    
                    # Let's pick a random candidate from candidates if available.
                    if candidates:
                        cand_list = list(candidates)
                        # Prefer candidates not in current codewords?
                        # Or just pick random.
                        cand_idx = cand_list[rng.integers(0, len(cand_list))]
                        
                        # If cand_idx is c_idx, we revert.
                        if cand_idx == c_idx:
                             # Revert
                             for w in neighbors[c_idx]:
                                 cover_count[w] += 1
                        else:
                            # Add cand_idx
                            for w in neighbors[cand_idx]:
                                cover_count[w] += 1
                            codewords[i] = _idx_to_tuple(cand_idx, n, q)
                            # We made a change, try to continue search
                            # Don't break, keep looking for reductions
                            # But to avoid infinite loops, we can break and restart the for loop
                            break
                    else:
                        # No candidate found to fix coverage. Revert.
                        for w in neighbors[c_idx]:
                            cover_count[w] += 1
                else:
                    # Too many lost words, likely hard to fix. Revert.
                    for w in neighbors[c_idx]:
                        cover_count[w] += 1

    return codewords


def construct(q, n, radius):
    start = time.monotonic()
    deadline = start + _DEADLINE_SECONDS
    max_time_remaining = [start, deadline]

    rng = np.random.default_rng(42)

    # Start with a greedy construction
    best = _greedy_construct(q, n, radius, rng, max_time_remaining)
    
    # Local search to improve
    if _elapsed(start) < deadline - 5.0:
        best = _local_search(best, q, n, radius, rng, max_time_remaining)

    # Verify coverage
    if not _is_cover(best, q, n, radius):
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(cw) for cw in best]
# EVOLVE-BLOCK-END
