"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return all primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _compute_score(primes, residues, max_check=100000):
    """Compute the largest L such that every integer 1..L has coverage >= 2."""
    n = len(primes)
    # Use a coverage array up to some reasonable bound
    # We need to find the first position where coverage < 2
    # Use difference array for efficiency
    bound = max_check
    cov = [0] * (bound + 2)
    
    for p, r in zip(primes, residues):
        # Mark all positions x where x ≡ r (mod p), x >= 1
        # Start from the smallest x >= 1 with x ≡ r (mod p)
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p
        for x in range(start, bound + 1, p):
            cov[x] += 1
    
    # Find the largest L such that cov[x] >= 2 for all x in 1..L
    L = 0
    for x in range(1, bound + 1):
        if cov[x] >= 2:
            L = x
        else:
            break
    
    return L

def _compute_score_fast(primes, residues, check_range):
    """Faster score computation for a limited range."""
    bound = check_range
    cov = [0] * (bound + 2)
    
    for p, r in zip(primes, residues):
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p
        for x in range(start, bound + 1, p):
            cov[x] += 1
    
    L = 0
    for x in range(1, bound + 1):
        if cov[x] >= 2:
            L = x
        else:
            break
    
    return L

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    primes = _sieve_primes(1000)
    n = len(primes)
    
    # Initial configuration: seed pattern
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Compute initial score
    current_score = _compute_score(primes, residues, 100000)
    best_residues = residues[:]
    best_score = current_score
    
    # Local search / hill climbing
    random.seed(42)
    
    # First, try simple hill climbing
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n):
            if time.time() > deadline - 10:
                break
            p = primes[i]
            old_r = residues[i]
            # Try all possible residues for this prime
            for new_r in range(p):
                if new_r == old_r:
                    continue
                residues[i] = new_r
                # Quick check: only need to check up to current_score + some margin
                check_bound = min(100000, current_score + 500)
                new_score = _compute_score_fast(primes, residues, check_bound)
                if new_score > current_score:
                    # Full verification
                    full_score = _compute_score(primes, residues, 100000)
                    if full_score > current_score:
                        current_score = full_score
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        improved = True
                        break
                    else:
                        residues[i] = old_r
                        break
                else:
                    residues[i] = old_r
                    break
            if time.time() > deadline - 10:
                break
    
    # If we still have time, try more aggressive search
    if time.time() < deadline - 5:
        # Simulated annealing style: try random mutations
        temp = 1000.0
        for iteration in range(5000):
            if time.time() > deadline - 5:
                break
            # Pick a random prime to mutate
            i = random.randint(0, n - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            residues[i] = new_r
            check_bound = min(100000, current_score + 500)
            new_score = _compute_score_fast(primes, residues, check_bound)
            
            # Accept if better, or with probability based on temperature
            if new_score > current_score:
                full_score = _compute_score(primes, residues, 100000)
                if full_score > current_score:
                    current_score = full_score
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues[:]
                else:
                    residues[i] = old_r
            else:
                # Reject, revert
                residues[i] = old_r
            
            # Cool down
            temp *= 0.999
    
    return best_residues

# EVOLVE-BLOCK-END
