import time
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

def construct():
    in_set = [False] * (N + 1)
    current_len = 0
    
    primes = _primes(N + 1)
    is_prime = [False] * (N + 2)
    for p in primes:
        is_prime[p] = True
        
    # Initialize covered structures
    # covered[p] is a bytearray of size p, 1 if residue is covered
    # covered_count[p] is the number of covered residues
    # missed_count[p] is p - covered_count[p]
    covered = {}
    covered_count = [0] * (N + 2)
    missed_count = [0] * (N + 2)
    # Store the missed residue if count is 1, for quick access
    # missed_res[p] is the residue if missed_count[p] == 1, else None
    missed_res = [None] * (N + 2)
    
    for p in primes:
        covered[p] = bytearray(p)
        covered_count[p] = 0
        missed_count[p] = p
        
    # Precompute multiples for fast marking?
    # Or just use slicing on numpy array?
    # blocked = np.zeros(N+1, dtype=bool)
    
    while time.time() < _DEADLINE:
        # Identify tight primes: those with missed_count == 1
        # and p <= current_len
        tight_primes = []
        for p in primes:
            if p > current_len:
                break
            if missed_count[p] == 1:
                # Find the missed residue
                # We can iterate 0..p-1
                # Optimization: maintain a list of missed residues?
                # For now, scan. p is small.
                miss_r = -1
                cov = covered[p]
                for r in range(p):
                    if not cov[r]:
                        miss_r = r
                        break
                if miss_r != -1:
                    missed_res[p] = miss_r
                    tight_primes.append((p, miss_r))
            elif missed_count[p] > 1:
                missed_res[p] = None # Not tight
            else:
                # Should not happen if we prevent violations
                break
        
        # Create blocked array
        blocked = [False] * (N + 1)
        
        # Mark blocked residues for tight primes
        for p, r in tight_primes:
            # x % p == r  =>  x = k*p + r
            # Mark all such x in [0, N]
            for x in range(r, N + 1, p):
                blocked[x] = True
        
        # Check p_new constraint
        p_new = current_len + 1
        if is_prime[p_new]:
            # We need to ensure S U {x} does not cover all residues mod p_new
            # i.e., there is some residue r' in p_new such that r' not in covered[p_new] and r' != x % p_new
            # Let missed_residues_new = {r' in [0, p_new) | not covered[p_new][r']}
            # If len(missed_residues_new) == 1, let r_miss be that one.
            # Then x % p_new must NOT be r_miss.
            # If len(missed_residues_new) > 1, any x is fine for this constraint?
            # No, if x covers one of the missed residues, the count decreases.
            # We just need count > 0 after adding.
            # Current missed count: m = missed_count[p_new]
            # If x % p_new is missed, new_m = m - 1.
            # If x % p_new is covered, new_m = m.
            # We need new_m > 0.
            # So if m == 1, x % p_new MUST be covered (i.e., not in missed set).
            # If m > 1, x can be anything.
            
            if missed_count[p_new] == 1:
                # Find the single missed residue
                r_miss = -1
                cov = covered[p_new]
                for r in range(p_new):
                    if not cov[r]:
                        r_miss = r
                        break
                if r_miss != -1:
                    # x % p_new != r_miss
                    # Mark x where x % p_new == r_miss as blocked
                    for x in range(r_miss, N + 1, p_new):
                        blocked[x] = True
            
        # Find first valid x
        best_x = -1
        for x in range(N + 1):
            if not in_set[x] and not blocked[x]:
                best_x = x
                break
        
        if best_x == -1:
            break
        
        # Commit best_x
        in_set[best_x] = True
        current_len += 1
        
        # Update state
        for p in primes:
            if p > N + 1:
                break
            r = best_x % p
            if not covered[p][r]:
                covered[p][r] = 1
                covered_count[p] += 1
                missed_count[p] -= 1

    result = [v for v in range(N + 1) if in_set[v]]
    return result
