import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 145.0

def _score(u):
    """Compute the score 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    n_sums, n_diffs = _compute_set_sizes_fft(u)
    if n_sums == 0:
        return 0.0
    ratio = n_diffs / float(n_sums)
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _compute_set_sizes_fft(u):
    """Compute |U+U| and |U-U| using FFT for exact counts."""
    if not u:
        return 0, 0
    m = max(u)
    size = 2 * m + 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    vec = np.zeros(fft_size, dtype=np.float64)
    for x in u:
        vec[x] = 1.0
    fft_vec = np.fft.fft(vec)
    sum_fft = fft_vec * fft_vec
    sum_conv = np.fft.ifft(sum_fft).real
    n_sums = int(np.sum(sum_conv[:size] > 0.5))
    diff_fft = fft_vec * np.conj(fft_vec)
    diff_conv = np.fft.ifft(diff_fft).real
    n_diffs = int(np.sum(diff_conv[:m+1] > 0.5))
    if m > 0:
        n_diffs += int(np.sum(diff_conv[fft_size-m:fft_size] > 0.5))
    return n_sums, n_diffs

def _score_fast(u_list):
    """Score using brute force for small sets, FFT for larger."""
    if not u_list or len(u_list) < 2:
        return 0.0
    m = max(u_list)
    if m < 1:
        return 0.0
    if m <= 300:
        sums = set()
        diffs = set()
        for a in u_list:
            for b in u_list:
                sums.add(a + b)
                diffs.add(a - b)
        n_sums = len(sums)
        n_diffs = len(diffs)
        if n_sums == 0:
            return 0.0
        ratio = n_diffs / float(n_sums)
        if ratio <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / math.log(2 * m + 1)
    else:
        return _score(u_list)

def _greedy_construct(n_target, M, seed=0):
    """
    Greedy constructive algorithm: build U one element at a time,
    always picking the candidate that maximizes the incremental score.
    """
    rng = random.Random(seed)
    u = [0]
    used = {0}
    
    # Pre-generate a pool of candidate positions to consider
    # Include structured candidates and random ones
    candidates_pool = set()
    # Structured: powers, multiples, etc.
    for i in range(1, M + 1):
        candidates_pool.add(i)
    # Add some random candidates
    for _ in range(min(2000, M * 2)):
        candidates_pool.add(rng.randint(1, M))
    candidates_pool.discard(0)
    
    for step in range(n_target - 1):
        if time.time() > _DEADLINE - 10.0:
            break
        
        best_x = -1
        best_score = -1.0
        # Limit candidates per step for speed
        if len(candidates_pool) > 500:
            cands = rng.sample(list(candidates_pool), 500)
        else:
            cands = list(candidates_pool)
        
        for x in cands:
            if x in used or x > M:
                continue
            new_u = u + [x]
            new_u = sorted(new_u)
            # Quick check: if x > M, skip (already filtered)
            s = _score_fast(new_u)
            if s > best_score:
                best_score = s
                best_x = x
        
        if best_x == -1:
            # No valid candidate found, break
            break
        
        u.append(best_x)
        u.sort()
        used.add(best_x)
        # Remove used from pool
        candidates_pool.discard(best_x)
    
    return u, _score_fast(u)

def _greedy_construct_randomized(n_target, M, seed=0, num_restarts=5):
    """
    Run greedy construction multiple times with different random seeds
    and candidate subsets to escape local optima.
    """
    best_u = None
    best_s = 0.0
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 10.0:
            break
        u, s = _greedy_construct(n_target, M, seed=seed + r * 1000)
        if s > best_s:
            best_s = s
            best_u = u[:]
    
    return best_u, best_s

def _local_refine(u, timeout=3.0):
    """Light local refinement: try moving each element to nearby positions."""
    u = sorted(set(u))
    if len(u) < 2:
        return u, _score_fast(u)
    
    current_score = _score_fast(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    n = len(u)
    max_val = max(u)
    
    improved = True
    while improved and time.time() < start_time + timeout and time.time() < _DEADLINE - 5.0:
        improved = False
        for i in range(n):
            if time.time() > start_time + timeout or time.time() > _DEADLINE - 5.0:
                break
            old_val = u[i]
            # Try a few nearby positions
            for delta in range(-max(1, max_val // 10), max(1, max_val // 10) + 1, max(1, max_val // 20)):
                new_val = old_val + delta
                if new_val < 0 or new_val > max_val * 2:
                    continue
                if new_val in u and new_val != old_val:
                    continue
                new_u = u[:]
                new_u[i] = new_val
                new_u = sorted(set(new_u))
                if len(new_u) != n:
                    continue
                s = _score_fast(new_u)
                if s > current_score + 1e-12:
                    u = new_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    max_val = max(u)
                    improved = True
                    break
    
    return best_u, best_score

def construct():
    global _DEADLINE
    random.seed(42)
    
    best_u = [0, 1]
    best_s = _score(best_u)
    
    # Strategy: Greedy constructive algorithm with different (n, M) pairs
    # Key insight: fix M (max element) to control the denominator,
    # then greedily add elements that maximize the ratio.
    
    # Phase 1: Try various (n, M) combinations with greedy construction
    # Smaller M means smaller denominator -> potentially higher score
    # But we need enough elements to make |U-U|/|U+U| large
    
    configs = [
        (n, M) for M in [50, 80, 100, 150, 200, 300, 400, 500]
        for n in range(4, min(M // 5 + 4, 30))
    ]
    
    # Sort by priority: prefer smaller M first (smaller denominator)
    configs.sort(key=lambda x: (x[1], -x[0]))
    
    # Deduplicate and limit
    seen = set()
    unique_configs = []
    for cfg in configs:
        if cfg not in seen:
            seen.add(cfg)
            unique_configs.append(cfg)
    
    # Try a subset of configs, prioritizing small M
    for n, M in unique_configs[:60]:
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Time budget per config
        time_budget = max(0.5, min(3.0, (_DEADLINE - time.time() - 10.0) / max(1, len(unique_configs) - unique_configs.index((n, M)))))
        
        u_opt, s_opt = _greedy_construct_randomized(n, M, seed=n * 100 + M, num_restarts=3)
        
        if u_opt and s_opt > best_s:
            best_s = s_opt
            best_u = u_opt[:]
    
    # Phase 2: Try some larger M with more elements
    for n, M in [(20, 500), (25, 600), (30, 700), (15, 400), (20, 300)]:
        if time.time() > _DEADLINE - 10.0:
            break
        
        u_opt, s_opt = _greedy_construct_randomized(n, M, seed=n * 1000 + M, num_restarts=3)
        
        if u_opt and s_opt > best_s:
            best_s = s_opt
            best_u = u_opt[:]
    
    # Phase 3: Local refinement on the best solution
    if time.time() < _DEADLINE - 5.0:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 1.0:
            u_opt, s_opt = _local_refine(best_u, timeout=min(remaining, 5.0))
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
    
    # Ensure 0 is in the set
    if 0 not in best_u:
        best_u = [0] + best_u
    
    return sorted(set(best_u))
