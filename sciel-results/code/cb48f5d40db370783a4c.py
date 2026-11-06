"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _primes_up_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.array(np.where(sieve)[0])

def _compute_L(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L is covered by >= 2 congruences.
    
    Uses an efficient approach: for each x, we need to count how many i satisfy x % p_i == r_i.
    We can precompute for each prime i, the set of x in [1, max_check] that are covered.
    Then for each x, sum the indicators.
    """
    n = len(primes)
    # For each prime i, compute which x in [1, max_check] satisfy x % p_i == r_i
    # x % p == r means x = r + k*p for k >= 0, and x >= 1
    # So x = max(r, 1) + k*p... actually x = r + k*p where r + k*p >= 1
    
    # Build a count array
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for i in range(n):
        p = primes[i]
        r = residues[i]
        # x values: r, r+p, r+2p, ... but only those >= 1 and <= max_check
        # If r == 0, x = p, 2p, 3p, ...
        # If r > 0, x = r, r+p, r+2p, ...
        if r == 0:
            start = p
        else:
            start = r
        for x in range(start, max_check + 1, p):
            counts[x] += 1
    
    # Find the largest L such that counts[1..L] are all >= 2
    L = 0
    for x in range(1, max_check + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _compute_L_fast(residues, primes, max_check=2000):
    """Fast evaluation for small max_check during optimization."""
    n = len(primes)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for i in range(n):
        p = primes[i]
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_check:
            counts[start::p] += 1
    
    L = 0
    for x in range(1, max_check + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _evaluate(residues, primes, max_check=2000):
    return _compute_L_fast(residues, primes, max_check)

def _greedy_improve(residues, primes, max_iter=3000, time_limit=8):
    """Greedy hill-climbing with plateau walking."""
    residues = np.array(residues, dtype=np.int64).copy()
    n = len(primes)
    best_L = _evaluate(residues, primes)
    start_time = time.time()
    
    no_improve_count = 0
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try changing each residue to a better value
        for i in range(n):
            p = primes[i]
            old_r = residues[i]
            
            # Try the current best and a few neighbors
            candidates = [old_r]
            # Add some random candidates
            for _ in range(2):
                candidates.append(np.random.randint(0, p))
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                residues[i] = new_r
                L = _evaluate(residues, primes)
                if L > best_L:
                    best_L = L
                    improved = True
                    break  # Keep this improvement
                else:
                    residues[i] = old_r
            
            if improved:
                no_improve_count = 0
                break
            else:
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 5:
                # Try a random perturbation to escape local optimum
                idx = np.random.randint(0, n)
                old_r = residues[idx]
                new_r = np.random.randint(0, primes[idx])
                residues[idx] = new_r
                L = _evaluate(residues, primes)
                if L < best_L:
                    residues[idx] = old_r
                else:
                    if L > best_L:
                        best_L = L
                    no_improve_count = 0
                if no_improve_count > 15:
                    break
    
    return residues, best_L

def construct():
    """Construct 168 residues, one for each prime p <= 1000 in increasing order."""
    primes = _primes_up_to(1000)
    n = len(primes)
    assert n == 168, f"Expected 168 primes, got {n}"
    
    deadline = time.time() + 140  # Safety margin
    
    # Start with seed
    seed_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    best_residues = seed_residues.copy()
    best_L = _evaluate(best_residues, primes)
    
    np.random.seed(42)
    
    # Multi-start
    for start in range(30):
        if time.time() > deadline:
            break
        
        if start == 0:
            residues = seed_residues.copy()
        elif start % 3 == 1:
            # Random init
            residues = np.array([np.random.randint(0, primes[i]) for i in range(n)], dtype=np.int64)
        else:
            # Perturb best
            residues = best_residues.copy()
            num_perturb = np.random.randint(3, 20)
            idxs = np.random.choice(n, size=num_perturb, replace=False)
            for idx in idxs:
                residues[idx] = np.random.randint(0, primes[idx])
        
        residues, L = _greedy_improve(residues, primes, max_iter=2000, time_limit=8)
        
        if L > best_L:
            best_L = L
            best_residues = residues.copy()
    
    # Final evaluation with larger max_check to get accurate score
    final_L = _compute_L(best_residues, primes, max_check=100000)
    
    return list(best_residues)
# EVOLVE-BLOCK-END
