"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000 (max sum is 2*10000)
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[i] == i:
        for j in range(i * i, _MAX_SUM + 1, i):
            if _sieve[j] == j:
                _sieve[j] = i

def _prime_factors_fast(n):
    """Return set of prime factors of n using precomputed sieve."""
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _compute_factor_sets(nums):
    """For each pair (i,j), compute the set of prime factors of nums[i]+nums[j].
    Return a list of 12 sets, where set[i] contains all prime factors of sums involving index i."""
    n = len(nums)
    pair_factors = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            pf = _prime_factors_fast(s)
            pair_factors[i][j] = pf
            pair_factors[j][i] = pf
    return pair_factors

def _total_factor_count(pair_factors):
    """Count distinct primes across all pair factor sets."""
    all_primes = set()
    n = len(pair_factors)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= pair_factors[i][j]
    return len(all_primes)

def _incremental_update(pair_factors, idx, old_val, new_val, nums):
    """Update pair_factors after changing nums[idx] from old_val to new_val.
    Returns the set of primes that were removed and added."""
    n = len(nums)
    removed = set()
    added = set()
    for j in range(n):
        if j == idx:
            continue
        # Old sum
        old_sum = old_val + nums[j]
        old_pf = _prime_factors_fast(old_sum)
        # New sum
        new_sum = new_val + nums[j]
        new_pf = _prime_factors_fast(new_sum)
        
        # Check if old primes are still present elsewhere
        for p in old_pf:
            still_present = False
            for i2 in range(n):
                for j2 in range(i2 + 1, n):
                    if (i2, j2) == (idx, j) or (i2, j2) == (j, idx):
                        continue
                    if p in pair_factors[i2][j2]:
                        still_present = True
                        break
                if still_present:
                    break
            if not still_present:
                removed.add(p)
        
        for p in new_pf:
            added.add(p)
        
        # Update the pair factor sets
        pair_factors[idx][j] = new_pf
        pair_factors[j][idx] = new_pf
    
    return removed, added

def construct():
    random.seed(12345)
    
    # Initial candidate: try several starting points
    # Strategy 1: multiples of a highly composite number
    # Strategy 2: small consecutive even numbers
    # Strategy 3: random
    
    candidates = []
    
    # Small even numbers
    candidates.append(list(range(2, 25, 2)))
    
    # Multiples of 6
    candidates.append(list(range(6, 78, 6)))
    
    # Multiples of 12
    candidates.append(list(range(12, 156, 12)))
    
    # Powers of 2 (not all distinct sums will be smooth)
    candidates.append([2**i for i in range(1, 13)])
    
    # Random starts
    for _ in range(5):
        candidates.append(sorted(random.sample(range(1, 10001), 12)))
    
    best_candidate = None
    best_score = float('inf')
    best_pair_factors = None
    
    for cand in candidates:
        if _deadline_hit():
            break
        if len(set(cand)) != 12:
            continue
        if not all(1 <= x <= 10000 for x in cand):
            continue
        pf = _compute_factor_sets(cand)
        score = _total_factor_count(pf)
        if score < best_score:
            best_score = score
            best_candidate = cand[:]
            best_pair_factors = pf
    
    if best_candidate is None:
        best_candidate = list(range(1, 13))
        best_pair_factors = _compute_factor_sets(best_candidate)
        best_score = _total_factor_count(best_pair_factors)
    
    # Local search with incremental updates
    # For speed, we'll do a simpler approach: try single-element changes
    # and use a faster scoring method
    
    current = best_candidate[:]
    current_pf = best_pair_factors
    current_score = best_score
    
    # Precompute: for each prime, how many pairs contain it
    # This helps with incremental updates
    
    n = 12
    
    def _get_prime_count(pf):
        """Count distinct primes."""
        primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                primes |= pf[i][j]
        return len(primes)
    
    # Hill climbing with random restarts
    for restart in range(200):
        if _deadline_hit():
            break
        
        # Try to improve current solution
        improved = True
        while improved and not _deadline_hit():
            improved = False
            # Try changing each element
            for idx in range(n):
                if _deadline_hit():
                    break
                old_val = current[idx]
                
                # Try several candidate values
                # Small perturbations
                candidates_to_try = set()
                for d in range(-50, 51):
                    v = old_val + d
                    if 1 <= v <= 10000 and v != old_val:
                        candidates_to_try.add(v)
                
                # Also try random values
                for _ in range(10):
                    candidates_to_try.add(random.randint(1, 10000))
                
                for new_val in candidates_to_try:
                    if new_val in current:
                        continue
                    
                    # Compute new score incrementally
                    # Only 11 sums change
                    new_primes = set()
                    for j in range(n):
                        if j == idx:
                            continue
                        s = new_val + current[j]
                        new_primes |= _prime_factors_fast(s)
                    
                    # Get primes from pairs not involving idx
                    other_primes = set()
                    for i in range(n):
                        if i == idx:
                            continue
                        for j in range(i + 1, n):
                            if j == idx:
                                continue
                            other_primes |= current_pf[i][j]
                    
                    total_primes = new_primes | other_primes
                    new_score = len(total_primes)
                    
                    if new_score < current_score:
                        # Accept
                        current[idx] = new_val
                        # Update pair_factors
                        for j in range(n):
                            if j != idx:
                                s = new_val + current[j]
                                current_pf[idx][j] = _prime_factors_fast(s)
                                current_pf[j][idx] = _prime_factors_fast(s)
                        current_score = new_score
                        improved = True
                        break
                
                if improved:
                    break
        
        # If stuck, try a random restart
        if not _deadline_hit():
            # Generate a new random candidate near the current best
            new_cand = current[:]
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 11)
                new_cand[idx] = random.randint(1, 10000)
            if len(set(new_cand)) == 12 and all(1 <= x <= 10000 for x in new_cand):
                new_pf = _compute_factor_sets(new_cand)
                new_score = _total_factor_count(new_pf)
                if new_score < current_score:
                    current = new_cand
                    current_pf = new_pf
                    current_score = new_score
    
    # Final verification
    result = current
    assert len(set(result)) == 12
    assert all(1 <= x <= 10000 for x in result)
    
    return result

# EVOLVE-BLOCK-END
