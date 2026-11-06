"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256

def _thue_morse_fast(n):
    """Generate Thue-Morse sequence: t(i) = popcount(i) % 2."""
    c = np.zeros(n, dtype=np.int8)
    for i in range(n):
        c[i] = bin(i).count('1') % 2
    return c

def _count_mono4_full(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.sum(mask)
    return int(count)

def _count_aps_containing(c, p):
    """Count monochromatic 4-term APs that include position p (0-indexed)."""
    count = 0
    # p can be at position 0, 1, 2, or 3 in the AP
    # AP: a, a+d, a+2d, a+3d where p = a + k*d for k in {0,1,2,3}
    
    # p is the first term: a = p, so p, p+d, p+2d, p+3d
    # Need p+3d < N => d <= (N-1-p)//3
    for d in range(1, (N - 1 - p) // 3 + 1):
        if c[p] == c[p+d] == c[p+2*d] == c[p+3*d]:
            count += 1
    
    # p is the second term: a+d = p => a = p-d, so p-d, p, p+d, p+2d
    # Need p-d >= 0 and p+2d < N => d <= min(p, (N-1-p)//2)
    for d in range(1, min(p, (N - 1 - p) // 2) + 1):
        if c[p-d] == c[p] == c[p+d] == c[p+2*d]:
            count += 1
    
    # p is the third term: a+2d = p => a = p-2d, so p-2d, p-d, p, p+d
    # Need p-2d >= 0 and p+d < N => d <= min(p//2, N-1-p)
    for d in range(1, min(p // 2, N - 1 - p) + 1):
        if c[p-2*d] == c[p-d] == c[p] == c[p+d]:
            count += 1
    
    # p is the fourth term: a+3d = p => a = p-3d, so p-3d, p-2d, p-d, p
    # Need p-3d >= 0 => d <= p//3
    for d in range(1, p // 3 + 1):
        if c[p-3*d] == c[p-2*d] == c[p-d] == c[p]:
            count += 1
    
    return count

def _delta_flip(c, p):
    """Compute change in mono-4AP count when flipping position p.
    Returns (delta, new_count_if_flipped) assuming current total is tracked externally.
    Actually returns the change: if we flip c[p], how many APs are gained/lost?
    
    An AP containing p is monochromatic before flip iff all 4 are same as c[p].
    After flip, c[p] becomes 1-c[p]. The AP is monochromatic after flip iff all other 3 are 1-c[p].
    
    So:
    - If AP was mono (all = c[p]): after flip, c[p] differs -> not mono. Loss of 1.
    - If AP was not mono but the other 3 are all = 1-c[p]: after flip, all = 1-c[p] -> mono. Gain of 1.
    - Otherwise: no change.
    """
    old_val = c[p]
    new_val = 1 - old_val
    delta = 0
    
    # p is the first term
    for d in range(1, (N - 1 - p) // 3 + 1):
        others = [c[p+d], c[p+2*d], c[p+3*d]]
        was_mono = (old_val == c[p+d] and old_val == c[p+2*d] and old_val == c[p+3*d])
        now_mono = (new_val == c[p+d] and new_val == c[p+2*d] and new_val == c[p+3*d])
        if was_mono and not now_mono:
            delta -= 1
        elif now_mono and not was_mono:
            delta += 1
    
    # p is the second term
    for d in range(1, min(p, (N - 1 - p) // 2) + 1):
        others = [c[p-d], c[p+d], c[p+2*d]]
        was_mono = (old_val == c[p-d] and old_val == c[p+d] and old_val == c[p+2*d])
        now_mono = (new_val == c[p-d] and new_val == c[p+d] and new_val == c[p+2*d])
        if was_mono and not now_mono:
            delta -= 1
        elif now_mono and not was_mono:
            delta += 1
    
    # p is the third term
    for d in range(1, min(p // 2, N - 1 - p) + 1):
        others = [c[p-2*d], c[p-d], c[p+d]]
        was_mono = (old_val == c[p-2*d] and old_val == c[p-d] and old_val == c[p+d])
        now_mono = (new_val == c[p-2*d] and new_val == c[p-d] and new_val == c[p+d])
        if was_mono and not now_mono:
            delta -= 1
        elif now_mono and not was_mono:
            delta += 1
    
    # p is the fourth term
    for d in range(1, p // 3 + 1):
        others = [c[p-3*d], c[p-2*d], c[p-d]]
        was_mono = (old_val == c[p-3*d] and old_val == c[p-2*d] and old_val == c[p-d])
        now_mono = (new_val == c[p-3*d] and new_val == c[p-2*d] and new_val == c[p-d])
        if was_mono and not now_mono:
            delta -= 1
        elif now_mono and not was_mono:
            delta += 1
    
    return delta

def _local_search_incremental(c, time_limit, random_seed=42):
    """Local search with incremental delta computation."""
    rng = random.Random(random_seed)
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_full(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        order = list(range(N))
        rng.shuffle(order)
        for pos in order:
            if (time.time() - start) >= time_limit:
                break
            delta = _delta_flip(best_c, pos)
            if delta < 0:
                best_c[pos] = 1 - best_c[pos]
                best_count += delta
                improved = True
    
    return best_c, best_count

def _simulated_annealing(c, time_limit, random_seed=42):
    """Simulated annealing with incremental delta computation."""
    rng = random.Random(random_seed)
    start = time.time()
    cur_c = c.copy()
    cur_count = _count_mono4_full(cur_c)
    best_c = cur_c.copy()
    best_count = cur_count
    
    # Temperature schedule
    T_start = 2.0
    T_end = 0.1
    # We'll estimate number of steps we can do
    # Each step: ~O(N) for delta computation
    # Let's just use time-based annealing
    n_steps_target = 50000  # rough estimate
    
    for step in range(n_steps_target):
        if (time.time() - start) >= time_limit:
            break
        # Temperature
        progress = step / n_steps_target
        T = T_start * (T_end / T_start) ** progress
        
        pos = rng.randint(0, N - 1)
        delta = _delta_flip(cur_c, pos)
        
        if delta <= 0:
            cur_c[pos] = 1 - cur_c[pos]
            cur_count += delta
            if cur_count < best_count:
                best_count = cur_count
                best_c = cur_c.copy()
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < np.exp(-delta / T):
                cur_c[pos] = 1 - cur_c[pos]
                cur_count += delta
                if cur_count < best_count:
                    best_count = cur_count
                    best_c = cur_c.copy()
    
    return best_c, best_count

def _two_flip_search(c, time_limit, random_seed=42):
    """Try flipping pairs of positions simultaneously."""
    rng = random.Random(random_seed)
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_full(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try random pairs
        for _ in range(2000):
            if (time.time() - start) >= time_limit:
                break
            i = rng.randint(0, N - 1)
            j = rng.randint(0, N - 1)
            if i == j:
                continue
            if best_c[i] == best_c[j]:
                continue  # Flipping both would be same as no change
            
            # Compute delta for flipping both i and j
            # Need to be careful: APs containing both i and j
            # Simple approach: flip i, compute delta_i, flip j, compute delta_j, but interactions...
            # For simplicity, just do sequential: flip i, then flip j, check total
            old_i = best_c[i]
            old_j = best_c[j]
            best_c[i] = 1 - old_i
            best_c[j] = 1 - old_j
            new_count = _count_mono4_full(best_c)
            if new_count < best_count:
                best_count = new_count
                improved = True
            else:
                best_c[i] = old_i
                best_c[j] = old_j
    
    return best_c, best_count

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    rng_global = random.Random(12345)
    
    # Generate diverse starting points
    starts = []
    
    # Thue-Morse and variants
    tm = _thue_morse_fast(N)
    starts.append(tm.copy())
    for offset in range(1, 16):
        starts.append(np.roll(tm, offset).copy())
    
    # Random with bias toward 50/50
    for _ in range(10):
        r = np.random.RandomState(rng_global.randint(0, 2**31))
        starts.append(r.randint(0, 2, N).astype(np.int8))
    
    # Alternating patterns
    for period in [2, 3, 4, 5, 6, 8]:
        pat = np.zeros(N, dtype=np.int8)
        for i in range(N):
            pat[i] = (i // period) % 2
        starts.append(pat.copy())
    
    best_c = None
    best_count = float('inf')
    
    # Evaluate all starts
    for s in starts:
        cnt = _count_mono4_full(s)
        if cnt < best_count:
            best_count = cnt
            best_c = s.copy()
    
    # Phase 1: Local search on best start
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 10, 30), 42)
    
    # Phase 2: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 15:
        sa_c, sa_count = _simulated_annealing(best_c, min(remaining - 15, 40), 777)
        if sa_count < best_count:
            best_count = sa_count
            best_c = sa_c
    
    # Phase 3: Local search again after SA
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 10, 25), 999)
    
    # Phase 4: Try local search on other good starts
    remaining = deadline - time.time()
    if remaining > 15:
        # Sort starts by their full count
        scored_starts = []
        for s in starts:
            cnt = _count_mono4_full(s)
            scored_starts.append((cnt, s))
        scored_starts.sort()
        
        for cnt, s in scored_starts[:5]:
            if time.time() >= deadline - 10:
                break
            if cnt >= best_count:
                continue
            ls_c, ls_count = _local_search_incremental(s, min(10, deadline - time.time() - 5), 555)
            if ls_count < best_count:
                best_count = ls_count
                best_c = ls_c
    
    # Phase 5: Final local search
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 5, 20), 111)
    
    # Phase 6: One more SA pass
    remaining = deadline - time.time()
    if remaining > 10:
        sa_c, sa_count = _simulated_annealing(best_c, min(remaining - 10, 20), 222)
        if sa_count < best_count:
            best_count = sa_count
            best_c = sa_c
    
    # Final local search
    remaining = deadline - time.time()
    if remaining > 3:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 3, 10), 333)
    
    # Convert to list of ints
    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
# EVOLVE-BLOCK-END
