"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes for primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _compute_score(residues, primes, max_check=100000):
    """
    Compute the largest L such that every integer 1..L satisfies at least
    two of the congruences x = r_p (mod p).
    Returns (L, coverage_array) where coverage_array[x] = count of residues hit.
    """
    n = len(primes)
    # coverage[x] = number of primes p such that x % p == r_p
    # We need to compute this efficiently.
    # For each prime p with residue r, the covered numbers in [1, max_check] are:
    # r, r+p, r+2p, ... (but r=0 means multiples of p)
    # Actually x = r mod p means x ≡ r (mod p), so x = r + k*p for k>=0, but x>=1.
    
    # Efficient approach: for each prime, mark the positions.
    # But with 168 primes and max_check up to 100000, we can use numpy.
    
    cov = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        # Numbers x in [1, max_check] where x % p == r
        # First such number >= 1: if r == 0, first is p; else first is r (if r>=1) or p (if r==0)
        if r == 0:
            start = p
        else:
            start = r
        if start > max_check:
            continue
        # Mark start, start+p, start+2p, ...
        indices = np.arange(start, max_check + 1, p)
        cov[indices] += 1
    
    # Find largest L such that cov[1..L] >= 2 for all
    # Find first position where cov < 2
    if max_check < 1:
        return 0, cov
    bad = np.where(cov[1:max_check+1] < 2)[0]
    if len(bad) == 0:
        return max_check, cov
    else:
        first_bad = bad[0] + 1  # 1-indexed
        return first_bad - 1, cov

def _find_bottleneck(cov, current_L):
    """Find the first integer > current_L that has coverage < 2, or None if all good up to some limit."""
    # We need to extend coverage. Let's check beyond current_L.
    # Actually, let's just return the first x >= 1 where cov[x] < 2
    # But we want the first x > current_L... no, we want the first x where cov[x] < 2
    # since L is defined as the largest such that all 1..L have cov >= 2.
    # So the bottleneck is the first x where cov[x] < 2.
    # If current_L is the score, then cov[current_L+1] < 2 (or current_L == max_check).
    # Let's just find the first x >= 1 with cov[x] < 2.
    if len(cov) < 2:
        return None
    bad = np.where(cov[1:] < 2)[0]
    if len(bad) == 0:
        return None
    return bad[0] + 1  # 1-indexed

def _coverage_at(x, residues, primes):
    """Count how many primes p have x % p == r_p."""
    count = 0
    for r, p in zip(residues, primes):
        if x % p == r:
            count += 1
    return count

def _try_fix_bottleneck(x, residues, primes, rng):
    """
    Try to fix the bottleneck at x by reassigning a residue of a prime that doesn't cover x.
    Returns (new_residues, success) or (None, False).
    """
    n = len(primes)
    # Find primes that do NOT cover x
    non_covering = [i for i in range(n) if x % primes[i] != residues[i]]
    if not non_covering:
        return None, False
    
    # Try random order of non-covering primes
    rng.shuffle(non_covering)
    
    for i in non_covering:
        p = primes[i]
        # We need to change residues[i] to some new_r such that x % p == new_r
        # i.e., new_r = x % p
        new_r = x % p
        if new_r == residues[i]:
            continue  # This would be the same, but we already checked it doesn't cover x
        
        # Try this assignment
        old_r = residues[i]
        residues[i] = new_r
        
        # Check if this maintains coverage for all 1..L (where L was the old score)
        # Actually, we need to check that no x' in [1, old_L] now has coverage < 2.
        # But changing one residue can only decrease coverage at positions where old_r was hit.
        # Let's compute the new score.
        # For efficiency, let's just recompute the score.
        # But that's expensive. Let's do a quick check: the only positions that could lose coverage
        # are those x' where x' % p == old_r. For those, coverage decreases by 1.
        # We need to make sure none of them drop below 2.
        
        # Quick check: for positions x' where x' % p == old_r and x' != x (since x now gains coverage),
        # check if coverage was exactly 2 (would drop to 1).
        # This is still O(max_check/p) per prime.
        
        # Let's just compute the score for now. It's 168 primes, should be fast enough.
        L_new, cov_new = _compute_score(residues, primes)
        
        if L_new >= 0:  # always true
            # Check if it's an improvement
            # Actually we just want to know if it's valid (no gap in prefix)
            # The score function already handles this.
            return list(residues), True
        else:
            residues[i] = old_r
    
    residues[i] = old_r  # restore (though i might be undefined if loop didn't run)
    return None, False

def _local_search(residues, primes, rng, time_limit):
    """
    Targeted local search: repeatedly find the bottleneck and fix it.
    """
    start_time = time.time()
    current_L, cov = _compute_score(residues, primes)
    
    while time.time() - start_time < time_limit:
        # Find bottleneck
        x_bottleneck = _find_bottleneck(cov, current_L)
        if x_bottleneck is None:
            break
        
        # Try to fix it
        new_residues, success = _try_fix_bottleneck(x_bottleneck, residues, primes, rng)
        if success:
            residues = new_residues
            current_L, cov = _compute_score(residues, primes)
        else:
            # Try random restart of local search or just break
            # Let's try a few random perturbations
            improved = False
            for _ in range(10):
                # Randomly change a residue
                i = rng.randint(0, len(primes) - 1)
                old_r = residues[i]
                residues[i] = rng.randint(0, primes[i] - 1)
                L_new, cov_new = _compute_score(residues, primes)
                if L_new > current_L:
                    current_L = L_new
                    cov = cov_new
                    improved = True
                    break
                else:
                    residues[i] = old_r
            
            if not improved:
                break
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)  # 168
    
    rng = random.Random(42)
    
    # Initial solution: use the seed strategy as a starting point
    # The seed: ((i + 2) // 2) % p for i, p in enumerate(primes)
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Compute initial score
    current_L, cov = _compute_score(residues, primes)
    
    best_L = current_L
    best_residues = list(residues)
    
    # Local search
    while time.time() < deadline:
        residues, current_L = _local_search(residues, primes, rng, time_limit=10)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = list(residues)
        
        # If we've reached a high score, we can stop
        if best_L >= 1000:
            break
        
        # Random restart: perturb the best solution
        residues = list(best_residues)
        # Perturb a few residues
        for _ in range(rng.randint(1, 5)):
            i = rng.randint(0, n_primes - 1)
            residues[i] = rng.randint(0, primes[i] - 1)
    
    return best_residues
# EVOLVE-BLOCK-END
