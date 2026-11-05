"""Simulated annealing for the ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock


def _tetracode_words():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _initial_codewords(q, n):
    # Direct sum of ternary tetracode with all words on remaining coords: 9 * 3**(n-4).
    tetracode = _tetracode_words()
    rest = n - 4
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=rest):
            words.append(h + list(tail))
    return words


def _cover_array_fast(codewords, q, n, radius):
    """Compute coverage counts via vectorized numpy where possible."""
    m = len(codewords)
    total = q ** n
    if m == 0:
        return np.zeros(total, dtype=np.int32)

    cd = np.array(codewords, dtype=np.int64)  # (m, n)
    cov = np.zeros(total, dtype=np.int64)

    powq = np.array([q ** (n - 1 - p) for p in range(n)], dtype=np.int64)
    # base index of each codeword
    idx = cd.dot(powq)  # (m,)

    for r in range(0, radius + 1):
        if r == 0:
            np.add.at(cov, idx, 1)
            continue
        for pos in itertools.combinations(range(n), r):
            pos_arr = np.array(pos, dtype=np.int64)
            others = np.array([p for p in range(n) if p not in pos], dtype=np.int64)
            # base index with pos positions zeroed
            base_idx = cd[:, others].dot(powq[others])  # (m,)
            contrib = powq[pos_arr]  # (r,)
            # iterate all q**r assignments
            for assign in itertools.product(range(q), repeat=r):
                assign_arr = np.array(assign, dtype=np.int64)
                offsets = assign_arr.dot(contrib)
                targets = base_idx + offsets
                np.add.at(cov, targets, 1)
    return cov.astype(np.int32)


def _greedy_reduce(codewords, q, n, radius, time_limit=5.0):
    """Remove redundant codewords while coverage is maintained."""
    if not codewords:
        return codewords
    t0 = time.time()
    cov = _cover_array_fast(codewords, q, n, radius)
    changed = True
    while changed and (time.time() - t0) < time_limit:
        changed = False
        for i in range(len(codewords)):
            if (time.time() - t0) >= time_limit:
                break
            # tentatively remove codeword i
            cov_i = _cover_array_fast([codewords[i]], q, n, radius)
            new_cov = cov - cov_i
            if np.all(new_cov >= 1):
                cov = new_cov
                codewords.pop(i)
                changed = True
                break
    return codewords


def _local_search(codewords, q, n, radius, time_limit=2.0, rng=None):
    """Iterative local search with critical-word guided moves."""
    if rng is None:
        rng = np.random.default_rng()
    t0 = time.time()
    m = len(codewords)
    
    # Precompute individual coverage arrays
    ind = [_cover_array_fast([c], q, n, radius) for c in codewords]
    cov_total = np.sum(np.stack(ind), axis=0) if ind else np.zeros(q ** n, dtype=np.int32)
    
    best_len = m
    best_cw = list(codewords)
    
    while time.time() - t0 < time_limit:
        # Find removable codewords (those that don't cause uncovered words)
        removable = []
        for i in range(m):
            if np.all(cov_total - ind[i] >= 1):
                removable.append(i)
        
        if removable:
            # Remove a random removable codeword
            i = int(rng.integers(0, len(removable)))
            idx = removable[i]
            cov_total = cov_total - ind[idx]
            codewords.pop(idx)
            ind.pop(idx)
            m -= 1
            if m < best_len:
                best_len = m
                best_cw = list(codewords)
        else:
            # Try to find a better replacement for a critical codeword
            # A codeword is critical if removing it creates uncovered words
            critical = []
            for i in range(m):
                if not np.all(cov_total - ind[i] >= 1):
                    critical.append(i)
            
            if not critical:
                break
            
            # Pick a random critical codeword
            i = int(rng.integers(0, len(critical)))
            idx = critical[i]
            
            # Try to find a neighbor that covers the same critical words
            best_neighbor = None
            best_gain = 0
            for _ in range(20):  # try 20 random neighbors
                new_word = list(codewords[idx])
                p = int(rng.integers(0, n))
                new_val = int(rng.integers(0, q))
                if new_val == new_word[p]:
                    continue
                new_word[p] = new_val
                
                # Check if this word is already in the set
                if any(c == new_word for c in codewords):
                    continue
                
                cov_new = _cover_array_fast([new_word], q, n, radius)
                # Words that would become uncovered if we remove old and add new
                newly_uncovered = np.sum((cov_total == 1) & (ind[idx] == 1) & (cov_new == 0))
                # Words that would become covered
                newly_covered = np.sum((cov_total == 0) & (cov_new == 1))
                
                if newly_uncovered == 0:
                    gain = newly_covered
                    if gain > best_gain:
                        best_gain = gain
                        best_neighbor = new_word
            
            if best_neighbor is not None:
                cov_total = cov_total - ind[idx] + _cover_array_fast([best_neighbor], q, n, radius)
                codewords[idx] = best_neighbor
                ind[idx] = _cover_array_fast([best_neighbor], q, n, radius)
            else:
                # No improvement found, kick: randomly modify a codeword
                if m > 0:
                    i = int(rng.integers(0, m))
                    codewords[i] = [int(rng.integers(0, q)) for _ in range(n)]
                    ind[i] = _cover_array_fast([codewords[i]], q, n, radius)
                    cov_total = np.sum(np.stack(ind), axis=0)
        
        # Check if we found a better solution
        if np.all(cov_total >= 1) and m < best_len:
            best_len = m
            best_cw = list(codewords)
    
    return best_cw


def construct(q, n, radius):
    global _DEADLINE
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Start from the tetracode construction (known valid covering).
    codewords = _initial_codewords(q, n)
    best_codewords = list(codewords)
    best_len = len(codewords)

    # Initial coverage check to ensure validity
    cov = _cover_array_fast(codewords, q, n, radius)
    if not np.all(cov >= 1):
        # Should not happen for the tetracode construction, but be safe.
        return best_codewords

    # Greedy reduction first to get a smaller starting point.
    codewords = _greedy_reduce(codewords, q, n, radius, time_limit=3.0)
    if len(codewords) < best_len:
        best_len = len(codewords)
        best_codewords = list(codewords)

    # ILS loop: local search + kick + local search
    while time.time() < _DEADLINE - 5.0:
        # Local search
        result = _local_search(list(codewords), q, n, radius, time_limit=2.0, rng=rng)
        if len(result) < best_len:
            best_len = len(result)
            best_codewords = list(result)
        
        # Check if we still have time
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Kick: randomly modify a few codewords
        codewords = list(result)
        if codewords:
            num_kicks = int(rng.integers(1, 4))
            for _ in range(num_kicks):
                i = int(rng.integers(0, len(codewords)))
                # Change 1-2 positions
                for _ in range(int(rng.integers(1, 3))):
                    p = int(rng.integers(0, n))
                    codewords[i][p] = int(rng.integers(0, q))
        
        # Verify validity after kick; if invalid, revert to best
        cov_kick = _cover_array_fast(codewords, q, n, radius)
        if not np.all(cov_kick >= 1):
            codewords = list(best_codewords)
    
    # Final greedy reduction on the best valid solution.
    reduced = _greedy_reduce(list(best_codewords), q, n, radius, time_limit=3.0)
    if len(reduced) < best_len:
        best_len = len(reduced)
        best_codewords = reduced

    # Ensure the best is valid.
    final_cov = _cover_array_fast(best_codewords, q, n, radius)
    if not np.all(final_cov >= 1):
        # Fallback to the initial tetracode construction.
        return _initial_codewords(q, n)

    return best_codewords
# EVOLVE-BLOCK-END
