"""Rewritten program: multi-start local search with efficient delta computation."""
import numpy as np
import time

N = 256
DEADLINE = 115.0

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N (0-indexed)."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _thue_morse():
    """Generate Thue-Morse sequence for 0..255."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_ap_membership():
    """Precompute which APs contain each position.
    Returns list of lists: ap_list[i] = list of (a, d, k) where position i is in AP at index k."""
    ap_list = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        for a in range(max_a):
            for k in range(4):
                pos = a + k * d
                ap_list[pos].append((a, d, k))
    return ap_list

def _delta_flip(c, i, ap_list):
    """Compute change in mono-4AP count if we flip position i, using precomputed membership."""
    delta = 0
    for (a, d, k) in ap_list[i]:
        idxs = [a, a + d, a + 2 * d, a + 3 * d]
        vals = [c[j] for j in idxs]
        was_mono = (vals[0] == vals[1] == vals[2] == vals[3])
        # After flip: position i has flipped
        if k == 0:
            new_vals = [1 - c[a], c[a+d], c[a+2*d], c[a+3*d]]
        elif k == 1:
            new_vals = [c[a], 1 - c[a+d], c[a+2*d], c[a+3*d]]
        elif k == 2:
            new_vals = [c[a], c[a+d], 1 - c[a+2*d], c[a+3*d]]
        else:
            new_vals = [c[a], c[a+d], c[a+2*d], 1 - c[a+3*d]]
        is_mono = (new_vals[0] == new_vals[1] == new_vals[2] == new_vals[3])
        if was_mono and not is_mono:
            delta -= 1
        elif not was_mono and is_mono:
            delta += 1
    return delta

def _local_search(c, time_limit, ap_list, max_passes=50):
    """Improve coloring via single-flip local search using delta computation.
    Returns (c, count)."""
    start = time.time()
    current = _count_mono4(c)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(N):
            if time.time() - start > time_limit:
                return c, current
            delta = _delta_flip(c, i, ap_list)
            if delta < 0:
                c[i] = 1 - c[i]
                current += delta
                improved = True
        if time.time() - start > time_limit:
            return c, current
    return c, current

def _random_coloring(rng):
    """Generate a random 0/1 coloring."""
    return rng.integers(0, 2, size=N).astype(np.int8)

def _mutate(c, rng, rate=0.03):
    """Bit-flip mutation with given rate."""
    flips = rng.random(N) < rate
    c[flips] = 1 - c[flips]
    return c

def construct():
    start_time = time.time()
    
    # Precompute AP membership for fast delta computation
    ap_list = _precompute_ap_membership()
    
    # Start with Thue-Morse
    c = _thue_morse()
    best_c = c.copy()
    best_count = _count_mono4(c)
    
    # Local search from Thue-Morse
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        c, cnt = _local_search(c, remaining - 10, ap_list)
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()
    
    # Multi-start local search with random restarts
    rng = np.random.default_rng(42)
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Try multiple random starts
    num_restarts = 30
    while remaining > 15:
        # Create a random or mutated starting point
        start_type = rng.random()
        if start_type < 0.3:
            c = _random_coloring(rng)
        elif start_type < 0.6:
            c = best_c.copy()
            c = _mutate(c, rng, rate=0.1)
        else:
            c = _thue_morse()
            c = _mutate(c, rng, rate=0.05)
        
        # Local search
        time_for_ls = min(3.0, (remaining - 10) / max(1, num_restarts))
        c, cnt = _local_search(c, time_for_ls, ap_list)
        
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()
        
        remaining = DEADLINE - (time.time() - start_time)
    
    # Final polish: more thorough local search on best solution
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        c = best_c.copy()
        c, cnt = _local_search(c, remaining - 3, ap_list, max_passes=100)
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()
    
    return [int(x) for x in best_c]
