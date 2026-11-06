import time
import random
import numpy as np

def _score_bitset(mask):
    """Count distinct d > 0 such that exists a with a, a+d, a+2d all in the set represented by mask.
    mask is a Python int where bit i is set iff value (i+1) is in the set.
    """
    # Extract the set of values
    vals = []
    m = mask
    idx = 0
    while m:
        if m & 1:
            vals.append(idx + 1)
        m >>= 1
        idx += 1
    
    n = len(vals)
    if n < 3:
        return 0
    
    vals_set = set(vals)
    diffs = set()
    for i in range(n):
        a = vals[i]
        for j in range(i + 1, n):
            d = vals[j] - a
            if d > 0:
                c = a + 2 * d
                if c <= 512 and c in vals_set:
                    diffs.add(d)
    return len(diffs)

def _score_fast(S):
    """Score a set of 32 integers using bitset for fast membership."""
    mask = 0
    for v in S:
        mask |= (1 << (v - 1))
    return _score_bitset(mask)

def _score_incremental(S, removed, added):
    """Score after swapping removed -> added in set S (S contains removed, not added)."""
    new_S = set(S)
    new_S.discard(removed)
    new_S.add(added)
    return _score_fast(new_S)

def _local_search(current, deadline, rng):
    """Perform local search with simulated annealing."""
    N = 32
    MAXV = 512
    current_score = _score_fast(current)
    best = set(current)
    best_score = current_score
    
    # Precompute the mask for fast updates
    current_mask = 0
    for v in current:
        current_mask |= (1 << (v - 1))
    
    T = 15.0
    T_min = 0.1
    cooling = 0.9995
    
    all_vals = list(range(1, MAXV + 1))
    
    while time.time() < deadline:
        # Generate a move: swap one element
        to_remove = rng.choice(list(current))
        # Choose to_add from values not in current
        # Use a random value, but bias toward values that might create progressions
        to_add = rng.randint(1, MAXV)
        attempts = 0
        while to_add in current and attempts < 20:
            to_add = rng.randint(1, MAXV)
            attempts += 1
        if to_add in current:
            continue
        
        new_mask = current_mask & ~(1 << (to_remove - 1)) | (1 << (to_add - 1))
        new_score = _score_bitset(new_mask)
        
        if new_score > current_score:
            current.discard(to_remove)
            current.add(to_add)
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = set(current)
        else:
            delta = new_score - current_score
            if delta < 0 and rng.random() < np.exp(delta / T):
                current.discard(to_remove)
                current.add(to_add)
                current_mask = new_mask
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_score

def _greedy_improve(S, deadline, rng):
    """Try greedy swaps that improve the score."""
    current = set(S)
    current_score = _score_fast(current)
    best = set(current)
    best_score = current_score
    
    current_mask = 0
    for v in current:
        current_mask |= (1 << (v - 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for to_remove in list(current):
            if time.time() >= deadline:
                break
            # Try a few random additions
            for _ in range(10):
                to_add = rng.randint(1, 512)
                if to_add in current:
                    continue
                new_mask = current_mask & ~(1 << (to_remove - 1)) | (1 << (to_add - 1))
                new_score = _score_bitset(new_mask)
                if new_score > current_score:
                    current.discard(to_remove)
                    current.add(to_add)
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = set(current)
                    improved = True
                    break
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    N = 32
    MAXV = 512
    rng = random.Random(42)
    
    # Start with a structured initial set that has many progressions
    # The set {1,2,...,32} has many 3-term progressions
    best_overall = set(range(1, N + 1))
    best_overall_score = _score_fast(best_overall)
    
    # Also try some other structured starts
    structured_starts = [
        set(range(1, N + 1)),           # consecutive
        set(range(1, 2*N+1, 2)),        # evens up to 62
        set(range(1, 2*N+1, 2)),        # same
        set(range(2, 2*N+2, 2)),        # evens 2..64
        set(range(1, N*2+1, 2)),        # odds 1..63
        set(range(1, 33)),              # 1..32
    ]
    
    for start_set in structured_starts:
        if len(start_set) == N:
            s = _score_fast(start_set)
            if s > best_overall_score:
                best_overall_score = s
                best_overall = set(start_set)
    
    # Multiple restarts with local search
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Random initial set
        current = set(rng.sample(range(1, MAXV + 1), N))
        current_score = _score_fast(current)
        
        if current_score > best_overall_score:
            best_overall = set(current)
            best_overall_score = current_score
        
        # Run local search with a time slice
        slice_deadline = min(deadline, time.time() + 10.0)
        best, best_score = _local_search(current, slice_deadline, rng)
        
        if best_score > best_overall_score:
            best_overall = best
            best_overall_score = best_score
        
        # If we have time, do a greedy improvement pass on the best
        if time.time() < deadline - 5.0:
            g_best, g_score = _greedy_improve(best_overall, deadline - 2.0, rng)
            if g_score > best_overall_score:
                best_overall = g_best
                best_overall_score = g_score
    
    return sorted(list(best_overall))
