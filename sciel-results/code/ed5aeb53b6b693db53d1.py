"""Rewritten program: greedy + local search for double covering."""
import time
import random

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

PRIMES = _primes_up_to(1000)
N_PRIMES = len(PRIMES)

def _compute_L(residues, cap=100000):
    """Compute the largest L such that every x in 1..L is covered at least twice."""
    counts = [0] * (cap + 1)
    for p, r in zip(PRIMES, residues):
        # Mark all x in 1..cap where x ≡ r (mod p)
        start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        for x in range(start, cap + 1, p):
            counts[x] += 1
    # Find the largest L where all 1..L have count >= 2
    L = 0
    for x in range(1, cap + 1):
        if counts[x] < 2:
            break
        L = x
    return L

def _compute_L_fast(residues, cap=100000):
    """Faster version using numpy if available, else fallback."""
    try:
        import numpy as np
        counts = np.zeros(cap + 1, dtype=np.int32)
        for p, r in zip(PRIMES, residues):
            start = r
            if start < 1:
                start += ((1 - start + p - 1) // p) * p
            if start <= cap:
                idx = np.arange(start, cap + 1, p)
                counts[idx] += 1
        # Find first position where count < 2
        bad = np.where(counts[1:] < 2)[0]
        if len(bad) == 0:
            return cap
        return bad[0]
    except ImportError:
        return _compute_L(residues, cap)

def _greedy_construct():
    """Greedy construction: assign each prime's residue to maximize coverage of the current prefix."""
    # Start with a target prefix
    # We'll iteratively build residues
    residues = [0] * N_PRIMES
    cap = 2000  # working cap for greedy
    
    # For each prime, choose the residue that maximizes the number of x in 1..cap
    # that are covered by this prime AND are currently under-covered (count < 2)
    # But we need to be careful: we want to extend the doubly-covered prefix.
    
    # Alternative greedy: for each prime, pick residue that covers the most
    # integers in [1, cap] that are currently covered 0 or 1 times.
    
    counts = [0] * (cap + 1)
    for i, p in enumerate(PRIMES):
        best_r = 0
        best_score = -1
        for r in range(p):
            # Compute how many x in 1..cap with x ≡ r (mod p) have counts[x] < 2
            start = r
            if start < 1:
                start += ((1 - start + p - 1) // p) * p
            score = 0
            for x in range(start, cap + 1, p):
                if counts[x] < 2:
                    score += 1
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        # Update counts
        start = best_r
        if start < 1:
            start += ((1 - start + p - 1) // p) * start  # BUG: should be p
        # Fix:
        start = best_r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        for x in range(start, cap + 1, p):
            counts[x] += 1
    
    return residues

def _local_search(residues, max_time=100.0, cap=10000):
    """Local search to improve the doubly-covered prefix length."""
    start_time = time.time()
    current = list(residues)
    current_L = _compute_L_fast(current, cap)
    best = list(current)
    best_L = current_L
    
    # If cap is too small, we may underestimate. Use a larger cap for evaluation.
    eval_cap = min(100000, max(20000, current_L * 3 + 1000))
    current_L = _compute_L_fast(current, eval_cap)
    best_L = current_L
    
    random.seed(42)
    
    while time.time() - start_time < max_time:
        # Pick a random prime index to change
        i = random.randrange(N_PRIMES)
        p = PRIMES[i]
        old_r = current[i]
        
        # Try a few random residues
        improved = False
        for _ in range(5):
            new_r = random.randrange(p)
            if new_r == old_r:
                continue
            current[i] = new_r
            L = _compute_L_fast(current, eval_cap)
            if L > current_L:
                current_L = L
                if L > best_L:
                    best_L = L
                    best = list(current)
                improved = True
                break
            else:
                current[i] = old_r
        
        # Occasionally try a larger eval cap
        if not improved and random.random() < 0.05:
            # Recompute with a larger cap to see if we can extend further
            larger_cap = min(100000, current_L + 5000)
            L_check = _compute_L_fast(current, larger_cap)
            if L_check > current_L:
                current_L = L_check
                eval_cap = larger_cap
                if L_check > best_L:
                    best_L = L_check
                    best = list(current)
    
    return best, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with greedy
    residues = _greedy_construct()
    
    # Quick evaluation
    L = _compute_L_fast(residues, 10000)
    
    # Local search
    residues, L = _local_search(residues, max_time=120.0)
    
    # Final check with full cap
    final_L = _compute_L_fast(residues, 100000)
    
    return residues

# EVOLVE-BLOCK-END
