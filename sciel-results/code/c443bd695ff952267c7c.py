"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_primes():
    """Return list of primes <= 1000."""
    sieve = [True] * 1001
    sieve[0] = sieve[1] = False
    for i in range(2, int(1000**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(2, 1001) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    primes = _get_primes()
    n = len(primes)
    
    # Max L to check
    MAX_L = 100000
    
    # Initial assignment
    residues = [i % p for i, p in enumerate(primes)]
    
    # Build coverage array: coverage[x] = number of congruences satisfied by x
    # coverage is 1-indexed: coverage[1]..coverage[MAX_L]
    coverage = np.zeros(MAX_L + 1, dtype=np.int32)
    
    def build_coverage(residues):
        """Build the coverage array from scratch."""
        cov = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            # x % p == r means x = r, r+p, r+2p, ... but x >= 1
            # First x >= 1 with x % p == r: if r == 0, x = p; else x = r
            # Actually x % p == r, for x in 1..MAX_L
            # The values are: r, r+p, r+2p, ... but if r==0, start at p (since x>=1, and 0%p==0 but x=0 not in range)
            # Wait: x % p == r. For r=0: x = p, 2p, 3p, ...
            # For r>0: x = r, r+p, r+2p, ...
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_L:
                cov[start::p] += 1
        return cov
    
    def find_first_gap(cov):
        """Find the first x >= 1 where cov[x] < 2."""
        # cov[1:] has values for x=1..MAX_L
        bad = np.where(cov[1:] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return bad[0] + 1  # first bad x, so L = bad[0]
    
    coverage = build_coverage(residues)
    current_score = find_first_gap(coverage)
    
    # Local search
    improved = True
    pass_count = 0
    while improved and time.time() < deadline:
        improved = False
        pass_count += 1
        for i in range(n):
            if time.time() > deadline:
                break
            p = primes[i]
            old_r = residues[i]
            best_r = old_r
            best_s = current_score
            
            # Determine candidate residues to try
            # Try: current, and a set of others
            # For efficiency, try a limited set
            if p <= 50:
                candidates = list(range(p))  # try all
            else:
                # Try current, neighbors, and random
                candidates = set()
                candidates.add(old_r)
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)
                # Add random candidates
                num_random = min(30, p)
                for _ in range(num_random):
                    candidates.add(int(np.random.randint(0, p)))
                candidates = list(candidates)
            
            for r in candidates:
                if r == old_r:
                    continue
                if time.time() > deadline:
                    break
                # Update coverage: remove old, add new
                # Remove contribution of old_r
                if old_r == 0:
                    old_start = p
                else:
                    old_start = old_r
                if old_start <= MAX_L:
                    coverage[old_start::p] -= 1
                
                # Add contribution of new r
                if r == 0:
                    new_start = p
                else:
                    new_start = r
                if new_start <= MAX_L:
                    coverage[new_start::p] += 1
                
                # Find score
                s = find_first_gap(coverage)
                
                if s > best_s:
                    best_s = s
                    best_r = r
                    residues[i] = r
                else:
                    # Revert
                    # Remove new, add old
                    if r == 0:
                        new_start = p
                    else:
                        new_start = r
                    if new_start <= MAX_L:
                        coverage[new_start::p] -= 1
                    if old_r == 0:
                        old_start = p
                    else:
                        old_start = old_r
                    if old_start <= MAX_L:
                        coverage[old_start::p] += 1
                    residues[i] = old_r
            
            if best_r != old_r:
                current_score = best_s
                improved = True
    
    return residues
# EVOLVE-BLOCK-END
