"""
Optimized solution using NumPy for vectorized coverage computation.
Strategy:
1. Fast evaluation of the double-covered prefix length L using NumPy.
2. Greedy construction to initialize residues.
3. Intensive local search (hill climbing) where for each selected prime,
   we evaluate all possible residues and pick the one that maximizes L.
"""
import time
import numpy as np

# Precompute primes <= 1000
def _get_primes(limit=1000):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i:limit+1:i] = False
    return [i for i in range(limit + 1) if sieve[i]]

PRIMES = _get_primes(1000)
N_PRIMES = len(PRIMES) # 168

def _compute_L(residues, primes, max_check=100000):
    """
    Compute the largest L such that every integer 1..L is covered by >= 2 congruences.
    Uses NumPy for speed.
    """
    # We only need to check up to max_check.
    # Create an array of indices 1..max_check
    # To save memory/time, we can process in chunks or just use a reasonably sized array.
    # Given max_check=100000, an array of int8/int16 is small.
    
    # Optimization: We can stop early if we find a gap.
    # However, vectorized operations are fast. Let's compute coverage for the whole range.
    
    # coverage[x] = sum(1 for p, r in zip(primes, residues) if x % p == r)
    
    # We'll build the coverage array incrementally or use broadcasting?
    # Broadcasting 168 arrays of size 100000 might be heavy (16.8M ops).
    # Let's use a loop over primes but vectorized inner operation.
    
    coverage = np.zeros(max_check, dtype=np.int32)
    
    for i, p in enumerate(primes):
        r = residues[i]
        # Indices x in [1, max_check-1] (0-indexed array, so x is 1..max_check)
        # x % p == r
        # If r == 0, x is multiple of p.
        # If r > 0, x is r, r+p, ...
        
        # Find first index >= 0 (representing value 1) such that (idx+1) % p == r
        # val = idx + 1
        # val % p == r
        # idx = val - 1
        # val_min:
        if r == 0:
            # val must be multiple of p. Smallest val >= 1 is p.
            # idx = p - 1
            start_val = p
        else:
            # val = r is smallest positive integer with val % p == r (since r < p)
            start_val = r
        
        # Indices in coverage array:
        # start_idx = start_val - 1
        # step = p
        # end = max_check
        
        start_idx = start_val - 1
        if start_idx < max_check:
            coverage[start_idx::p] += 1
            
    # Find the first index where coverage < 2
    # We want the largest L such that coverage[0:L] >= 2
    # coverage is 0-indexed corresponding to values 1..max_check
    
    # Create mask of valid positions
    valid = coverage >= 2
    
    # Find the first False in valid
    # If all are True, L = max_check
    # If first False is at index k, L = k
    
    # np.argmax returns first non-zero (True) if we invert?
    # Let's find indices where valid is False
    # If no False, L = max_check
    # Else, L = first index of False
    
    # Note: L is the count of consecutive trues from the start.
    # If valid[0] is False, L=0.
    # If valid[0], valid[1] are True, valid[2] False, L=2.
    
    # np.where(valid == False)[0] gives indices of False.
    # But we only care about the *first* one.
    
    # Efficient way:
    # If not np.all(valid):
    #    first_false_idx = np.argmax(~valid)
    #    L = first_false_idx
    # else:
    #    L = max_check
    
    # However, we should also consider that the problem asks for 1..L.
    # If L=100, it means 1..100 are covered.
    
    if np.all(valid):
        return max_check
    
    first_false_idx = np.argmax(~valid)
    return int(first_false_idx)

def construct():
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 10.0
    deadline = start_time + time_limit - safety_margin
    
    # Initial greedy construction
    # For each prime, pick residue that maximizes coverage in a small window
    # We'll use a small window for init, e.g., 1000
    init_window = 1000
    residues = [0] * N_PRIMES
    
    # Fast init: just pick 0 for all? Or random?
    # Let's try a quick greedy pass.
    # For each prime p_i, try all r in 0..p_i-1 and pick best for L in init_window.
    # This is 168 * avg_p * cost_eval.
    # avg_p ~ 500. 168*500 = 84000 evals.
    # Each eval is ~168 numpy ops.
    # 84000 * 168 ~ 14M ops. Should be fast with numpy.
    
    for i, p in enumerate(PRIMES):
        best_r = 0
        best_L = -1
        # We can optimize by only checking a subset if it's slow, but let's try all.
        for r in range(p):
            residues[i] = r
            L = _compute_L(residues, PRIMES, max_check=init_window)
            if L > best_L:
                best_L = L
                best_r = r
        residues[i] = best_r
        
    # Local Search / Hill Climbing
    # We will iterate, picking a random prime, and finding the best residue for it.
    # To speed up, we can evaluate all residues for that prime at once?
    # _compute_L takes a full list. We can modify it to take a single prime update?
    # Or just call _compute_L for each candidate residue.
    # If we pick prime i, we try p_i candidates.
    # Cost per iteration: p_i * cost_eval.
    # p_i is up to 1000. cost_eval is ~168 numpy ops.
    # 1000 * 168 = 168,000 ops per iteration.
    # In 100s, we can do ~1000-5000 iterations?
    # Let's check speed of _compute_L.
    
    # Let's do a quick benchmark in head:
    # _compute_L does 168 slices.
    # Slice add is very fast.
    # So _compute_L is probably < 1ms.
    # 1000 candidates * 1ms = 1s per iteration.
    # We have ~110s. So ~100 iterations.
    # 100 iterations * 168 primes coverage = 16800 updates.
    # This might be enough to improve significantly.
    
    # To improve further, we can use a smaller window for search, e.g., 2000 or 5000.
    # The "true" L might be larger, but local improvements in the prefix are likely to extend it.
    # Let's use a window of 5000 for search to balance accuracy and speed.
    search_window = 5000
    
    current_residues = list(residues)
    current_L = _compute_L(current_residues, PRIMES, max_check=search_window)
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Pick a random prime index
        i = rng.integers(0, N_PRIMES)
        p = PRIMES[i]
        old_r = current_residues[i]
        
        best_r = old_r
        best_L = current_L
        
        # Try all residues for this prime
        # Optimization: We can break early if we find L == search_window?
        # But we want to maximize, so if we hit the cap, it's good.
        
        # To make this faster, we can sample residues if p is large?
        # No, trying all is better for local optimum.
        # But if p=997, that's 997 evals.
        
        # Let's try all.
        for r in range(p):
            if r == old_r:
                continue
            current_residues[i] = r
            L = _compute_L(current_residues, PRIMES, max_check=search_window)
            if L > best_L:
                best_L = L
                best_r = r
            # If we hit the cap, we can't do better in this window
            if best_L >= search_window:
                break
                
        current_residues[i] = best_r
        
        if best_L > current_L:
            current_L = best_L
            # If we improved, maybe continue with same prime or restart?
            # Standard hill climbing: accept and move to next random prime.
            
    # Final check with larger window to ensure we didn't overfit to search_window
    # But the score is based on the actual L, so if we extended the prefix in search_window,
    # it likely extends in the real domain too, unless the gap was just beyond search_window.
    # We should return the residues that maximize the *actual* L.
    # Our local search maximized L in search_window.
    # It's possible that a different set of residues gives a smaller L in search_window
    # but a larger L in the real domain (e.g. if search_window was too small and we stopped early).
    # However, usually extending the prefix locally helps globally.
    
    # Let's do a final polish with a larger window if time permits?
    # No, we used the deadline.
    
    return current_residues
