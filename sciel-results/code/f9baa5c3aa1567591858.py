"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _primes_upto(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_score(primes, residues):
    """Compute the largest L such that every integer 1..L is covered by >= 2 congruences.
    Returns (L, min_count_over_1_to_L)."""
    n = len(primes)
    max_L = 100000  # cap
    # We'll check progressively larger L
    # For efficiency, check up to some limit first
    # Use vectorized approach: for each x, count how many primes p have x % p == residues[i]
    
    # Start with a reasonable search range
    # Check L from 1 upward, but do it efficiently
    # We'll binary search or just scan
    
    # For a given L, we need to check x=1..L
    # Let's check up to 2000 first, then expand if needed
    check_limit = 2000
    
    x = np.arange(1, check_limit + 1)
    # For each prime i, create a boolean array where x % p_i == residues[i]
    # Count per x
    counts = np.zeros(check_limit, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        # x % p == r  =>  x is in {r, r+p, r+2p, ...} within [1, check_limit]
        # Vectorized: (x % p) == r
        counts += (x % p == r).astype(np.int32)
    
    # Find the largest L such that min(counts[0:L]) >= 2
    # i.e., the first position where count < 2
    below2 = np.where(counts < 2)[0]
    if len(below2) == 0:
        # All up to check_limit are covered >= 2
        # Need to check further
        # Extend check
        L = check_limit
        # Check further
        while L < max_L:
            next_limit = min(L + 1000, max_L)
            x2 = np.arange(L + 1, next_limit + 1)
            counts2 = np.zeros(len(x2), dtype=np.int32)
            for i, p in enumerate(primes):
                r = residues[i]
                counts2 += (x2 % p == r).astype(np.int32)
            below2_2 = np.where(counts2 < 2)[0]
            if len(below2_2) == 0:
                L = next_limit
            else:
                L = L + below2_2[0]
                break
        return L, 2
    else:
        # First position (0-indexed) where count < 2
        first_fail = below2[0]
        L = first_fail  # integers 1..first_fail are all >= 2, first_fail+1 is not
        # Actually x[0]=1, so if counts[0] < 2, L=0
        # If counts[k] < 2, then x=k+1 fails, so L = k
        L = first_fail
        min_c = int(np.min(counts[:L])) if L > 0 else 0
        return L, min_c

def _compute_score_fast(primes, residues, max_check=2000):
    """Faster score computation for SA inner loop."""
    x = np.arange(1, max_check + 1)
    counts = np.zeros(max_check, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        counts += (x % p == r).astype(np.int32)
    below2 = np.where(counts < 2)[0]
    if len(below2) == 0:
        return max_check
    return int(below2[0])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    primes = _primes_upto(1000)
    n = len(primes)  # 168
    assert n == 168, f"Expected 168 primes, got {n}"
    
    best_score = 0
    best_residues = None
    
    # Precompute x array for fast evaluation
    max_check = 2000
    x_arr = np.arange(1, max_check + 1)
    
    def eval_score(residues):
        """Quick score for SA: returns L (capped at max_check for speed)."""
        counts = np.zeros(max_check, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            counts += (x_arr % p == r).astype(np.int32)
        below2 = np.where(counts < 2)[0]
        if len(below2) == 0:
            return max_check
        return int(below2[0])
    
    def full_score(residues):
        """Full score with extension beyond max_check."""
        L = eval_score(residues)
        if L < max_check:
            return L
        # Extend
        cur_L = max_check
        while cur_L < 100000:
            next_L = min(cur_L + 500, 100000)
            x2 = np.arange(cur_L + 1, next_L + 1)
            counts2 = np.zeros(len(x2), dtype=np.int32)
            for i, p in enumerate(primes):
                r = residues[i]
                counts2 += (x2 % p == r).astype(np.int32)
            below2_2 = np.where(counts2 < 2)[0]
            if len(below2_2) == 0:
                cur_L = next_L
            else:
                cur_L = cur_L + int(below2_2[0])
                break
        return cur_L
    
    # Multiple restarts
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Initialize
        if restart == 0:
            # Greedy small-prime init: assign residues to cover small integers well
            residues = np.zeros(n, dtype=np.int64)
            # For each prime, try to set residue to cover a small number
            # Strategy: for prime p at index i, set r_p = (i+2) % p (like seed)
            for i, p in enumerate(primes):
                residues[i] = (i + 2) % p
        else:
            # Random init with some structure
            residues = np.zeros(n, dtype=np.int64)
            for i, p in enumerate(primes):
                # Bias toward covering small numbers
                target = rng.randint(1, min(p, 200))
                residues[i] = target % p
        
        # SA
        T_start = 50.0
        T_end = 0.1
        n_iter = 20000
        best_local = eval_score(residues)
        best_local_res = residues.copy()
        
        # Precompute primes array for vectorized use
        primes_arr = np.array(primes, dtype=np.int64)
        
        for it in range(n_iter):
            if time.time() > deadline - 10:
                break
            
            T = T_start * (T_end / T_start) ** (it / n_iter)
            
            # Pick random prime index
            i = rng.randint(0, n)
            p = primes[i]
            
            # Try a new random residue
            old_r = residues[i]
            new_r = rng.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Compute delta in score
            # The change affects counts for all x where x % p == old_r (decrease) and x % p == new_r (increase)
            # For speed, just re-evaluate
            residues[i] = new_r
            new_score = eval_score(residues)
            
            if new_score >= best_local:
                best_local = new_score
                best_local_res = residues.copy()
            else:
                # Accept with probability exp((new_score - best_local) / T)
                # But we want to accept moves that don't hurt too much
                delta = new_score - best_local  # negative
                if delta < 0:
                    prob = np.exp(delta / max(T, 0.01))
                    if rng.random() > prob:
                        residues[i] = old_r
                    else:
                        # Accept worse move
                        pass
                else:
                    # Shouldn't happen since new_score >= best_local handled above
                    pass
        
        # After SA, do targeted local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            cur_score = eval_score(residues)
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                p = primes[i]
                old_r = residues[i]
                best_r = old_r
                best_s = cur_score
                # Try all residues for this prime (only if p is small enough)
                if p <= 100:
                    for r in range(p):
                        if r == old_r:
                            continue
                        residues[i] = r
                        s = eval_score(residues)
                        if s > best_s:
                            best_s = s
                            best_r = r
                    residues[i] = best_r
                    if best_r != old_r:
                        improved = True
                        cur_score = best_s
        
        # Final full score
        final_s = full_score(residues)
        if final_s > best_score:
            best_score = final_s
            best_residues = residues.copy()
    
    # If no improvement found, use seed
    if best_residues is None or best_score < 84:
        best_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
        best_score = full_score(best_residues)
    
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
