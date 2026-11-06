"""Rewritten program: vectorized greedy + local search for double coverage."""
import numpy as np
import time

def _primes_up_to(n):
    """Return list of primes <= n."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.array([i for i in range(n + 1) if sieve[i]])

def _compute_L(residues, primes, L_max=100000):
    """Compute the largest L such that every integer 1..L is covered >= 2 times."""
    n = len(primes)
    x = np.arange(1, L_max + 1)
    coverage = np.zeros(L_max, dtype=np.int32)
    for i in range(n):
        p = primes[i]
        r = residues[i]
        # positions where x ≡ r (mod p), i.e., x = r + kp
        # x in [1, L_max], so r + kp >= 1 and r + kp <= L_max
        # kp >= 1 - r, kp <= L_max - r
        # k >= ceil((1-r)/p), k <= floor((L_max - r)/p)
        k_min = 0 if r >= 1 else 0  # since r >= 0, 1-r <= 1, so k_min = 0 always if r < p
        # Actually: x = r + kp. If r == 0, x = kp, first is p. If r >= 1, first is r.
        # Let's just use: positions = r + p * arange(k_min, k_max+1)
        # Simpler: use modulo
        mask = (x % p) == (r % p)
        coverage += mask.astype(np.int32)
    # Find largest L such that coverage[0:L] all >= 2
    # coverage[j] corresponds to x = j+1
    valid = coverage >= 2
    # Find first position where valid is False
    if not valid.any():
        return 0
    # Find the longest prefix of True
    # Use cumulative product or argmax on ~valid
    if valid.all():
        return L_max
    first_false = np.argmax(~valid)
    return first_false

def _compute_L_fast(residues, primes, L_max=100000):
    """Faster L computation using slice assignment."""
    n = len(primes)
    coverage = np.zeros(L_max, dtype=np.int32)
    for i in range(n):
        p = primes[i]
        r = residues[i]
        # x in [1, L_max] with x ≡ r (mod p)
        # x = r + kp. First valid x: if r >= 1 then r, else p (when r=0)
        # Actually r is in [0, p-1]. x = r + kp for k=0,1,2,...
        # x >= 1: if r >= 1, k=0 gives x=r >= 1. If r=0, k=0 gives x=0 (invalid), k=1 gives x=p.
        # x <= L_max: k <= (L_max - r) / p
        if r == 0:
            start = p  # first valid x
        else:
            start = r
        if start > L_max:
            continue
        # positions: start, start+p, start+2p, ...
        # In 0-indexed array (index = x-1): start-1, start-1+p, ...
        idx = start - 1
        coverage[idx::p] += 1
    valid = coverage >= 2
    if valid.all():
        return L_max
    first_false = np.argmax(~valid)
    return first_false

def _greedy_init(primes, L_max=20000):
    """Greedy construction: for each prime, pick residue maximizing coverage of under-covered spots."""
    n = len(primes)
    residues = np.zeros(n, dtype=np.int64)
    coverage = np.zeros(L_max, dtype=np.int32)
    
    for i in range(n):
        p = primes[i]
        best_r = 0
        best_score = -1
        # Try all residues for this prime
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            if start > L_max:
                score = 0
            else:
                idx = start - 1
                # Count how many of these positions currently have coverage < 2
                mask = coverage[idx::p] < 2
                score = mask.sum()
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        # Update coverage
        p = primes[i]
        r = best_r
        if r == 0:
            start = p
        else:
            start = r
        if start <= L_max:
            idx = start - 1
            coverage[idx::p] += 1
    
    return residues

def _local_search(residues, primes, L_max=100000, max_iter=5000, deadline=None):
    """Local search: try changing one residue at a time."""
    n = len(primes)
    best_L = _compute_L_fast(residues, primes, L_max)
    current = residues.copy()
    no_improve = 0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random prime to modify
        i = np.random.randint(n)
        p = primes[i]
        old_r = current[i]
        if p <= 2:
            # Only 1-2 choices, skip or try the other
            candidates = [r for r in range(p) if r != old_r]
        else:
            # Try a few random candidates
            candidates = np.random.choice(p, size=min(5, p-1), replace=False)
            candidates = [c for c in candidates if c != old_r]
        
        improved = False
        for new_r in candidates:
            current[i] = new_r
            L = _compute_L_fast(current, primes, L_max)
            if L > best_L:
                best_L = L
                improved = True
                no_improve = 0
                break
            else:
                current[i] = old_r
        if not improved:
            current[i] = old_r
            no_improve += 1
            if no_improve > 200:
                break
    
    return current, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    primes = _primes_up_to(1000)
    n = len(primes)
    
    best_residues = None
    best_L = 0
    
    # Greedy init
    residues = _greedy_init(primes, L_max=20000)
    L = _compute_L_fast(residues, primes, 100000)
    if L > best_L:
        best_L = L
        best_residues = residues.copy()
    
    # Local search from greedy
    if time.time() < deadline - 10:
        residues, L = _local_search(residues, primes, 100000, max_iter=10000, deadline=deadline)
        if L > best_L:
            best_L = L
            best_residues = residues.copy()
    
    # Multiple random restarts
    np.random.seed(42)
    restart = 0
    while time.time() < deadline - 15:
        # Random init with some structure
        residues = np.array([np.random.randint(p) for p in primes], dtype=np.int64)
        # Quick greedy improvement: for each prime, try to improve
        for i in range(n):
            p = primes[i]
            best_r = residues[i]
            best_L_i = _compute_L_fast(residues, primes, 10000)
            # Try a few random alternatives
            for _ in range(3):
                r = np.random.randint(p)
                if r == best_r:
                    continue
                residues[i] = r
                L_i = _compute_L_fast(residues, primes, 10000)
                if L_i > best_L_i:
                    best_r = r
                    best_L_i = L_i
            residues[i] = best_r
        
        L = _compute_L_fast(residues, primes, 100000)
        if L > best_L:
            best_L = L
            best_residues = residues.copy()
        
        # Local search from this start
        if time.time() < deadline - 10:
            residues, L = _local_search(residues, primes, 100000, max_iter=3000, deadline=deadline)
            if L > best_L:
                best_L = L
                best_residues = residues.copy()
        
        restart += 1
    
    if best_residues is None:
        # Fallback: seed construction
        best_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    
    return list(best_residues)
