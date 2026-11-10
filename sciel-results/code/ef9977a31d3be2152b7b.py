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

# Precompute prime factor sets for all possible sums 2..20000
_sum_pf = [None] * (_MAX_SUM + 1)
for s in range(2, _MAX_SUM + 1):
    _sum_pf[s] = _prime_factors_fast(s)

def _pf(s):
    """Fast lookup of prime factors for a sum."""
    return _sum_pf[s]

def _compute_factor_sets(nums):
    """For each pair (i,j), compute the set of prime factors of nums[i]+nums[j].
    Return a list of 12 lists, where pair_factors[i][j] is the set of prime factors."""
    n = len(nums)
    pair_factors = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            pf = _pf(s)
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

def _score_from_pf(pair_factors):
    """Quick score computation from pair_factors structure."""
    all_primes = set()
    n = len(pair_factors)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= pair_factors[i][j]
    return len(all_primes)

def construct():
    random.seed(12345)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Small even numbers
    candidates.append(list(range(2, 25, 2)))
    
    # Multiples of 6
    candidates.append(list(range(6, 78, 6)))
    
    # Multiples of 12
    candidates.append(list(range(12, 156, 12)))
    
    # Multiples of 4
    candidates.append(list(range(4, 52, 4)))
    
    # Multiples of 3
    candidates.append(list(range(3, 39, 3)))
    
    # Powers of 2
    candidates.append([2**i for i in range(1, 13)])
    
    # Small odd numbers
    candidates.append(list(range(1, 24, 2)))
    
    # Random starts
    for _ in range(10):
        candidates.append(sorted(random.sample(range(1, 10001), 12)))
    
    # Random starts with small values
    for _ in range(10):
        candidates.append(sorted(random.sample(range(1, 200), 12)))
    
    # Random starts with even values
    for _ in range(10):
        candidates.append(sorted(random.sample(range(2, 10000, 2), 12)))
    
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
        score = _score_from_pf(pf)
        if score < best_score:
            best_score = score
            best_candidate = cand[:]
            best_pair_factors = pf
    
    if best_candidate is None:
        best_candidate = list(range(1, 13))
        best_pair_factors = _compute_factor_sets(best_candidate)
        best_score = _score_from_pf(best_pair_factors)
    
    # Local search with efficient incremental updates
    current = best_candidate[:]
    current_pf = best_pair_factors
    current_score = best_score
    
    n = 12
    
    def _try_move(idx, new_val):
        """Try replacing current[idx] with new_val. Returns (new_score, new_pf) if improved, else None."""
        if new_val in current:
            return None
        if not (1 <= new_val <= 10000):
            return None
        
        # Compute new primes from pairs involving idx
        new_primes_idx = set()
        for j in range(n):
            if j == idx:
                continue
            s = new_val + current[j]
            new_primes_idx |= _pf(s)
        
        # Get primes from pairs not involving idx
        other_primes = set()
        for i in range(n):
            if i == idx:
                continue
            for j in range(i + 1, n):
                if j == idx:
                    continue
                other_primes |= current_pf[i][j]
        
        total_primes = new_primes_idx | other_primes
        new_score = len(total_primes)
        
        if new_score < current_score:
            # Create updated pair_factors
            new_pf = [row[:] for row in current_pf]
            for j in range(n):
                if j != idx:
                    s = new_val + current[j]
                    new_pf[idx][j] = _pf(s)
                    new_pf[j][idx] = _pf(s)
            return (new_score, new_pf)
        
        return None
    
    # Hill climbing with multiple restarts and diverse move strategies
    for restart in range(500):
        if _deadline_hit():
            break
        
        improved = True
        while improved and not _deadline_hit():
            improved = False
            
            # Try different move strategies
            for idx in range(n):
                if _deadline_hit():
                    break
                
                old_val = current[idx]
                
                # Strategy 1: Small perturbations
                for d in range(-100, 101):
                    v = old_val + d
                    if v == old_val:
                        continue
                    result = _try_move(idx, v)
                    if result:
                        current[idx] = v
                        current_score, current_pf = result
                        improved = True
                        break
                
                if improved:
                    break
                
                # Strategy 2: Random values
                for _ in range(20):
                    v = random.randint(1, 10000)
                    result = _try_move(idx, v)
                    if result:
                        current[idx] = v
                        current_score, current_pf = result
                        improved = True
                        break
                
                if improved:
                    break
                
                # Strategy 3: Values that make sums divisible by small primes
                for target_prime in [2, 3, 5, 7]:
                    for j in range(n):
                        if j == idx:
                            continue
                        # We want (v + current[j]) % target_prime == 0
                        # So v % target_prime == (-current[j]) % target_prime
                        remainder = (-current[j]) % target_prime
                        # Try a few values with this remainder
                        base = remainder
                        for offset in range(0, 10000, target_prime):
                            v = base + offset
                            if v == old_val or not (1 <= v <= 10000):
                                continue
                            if v in current:
                                continue
                            result = _try_move(idx, v)
                            if result:
                                current[idx] = v
                                current_score, current_pf = result
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if improved:
                    break
        
        # If stuck, try a random restart near current best
        if not _deadline_hit():
            new_cand = current[:]
            num_changes = random.randint(1, 4)
            for _ in range(num_changes):
                idx = random.randint(0, 11)
                # Bias towards small values
                if random.random() < 0.5:
                    new_cand[idx] = random.randint(1, 200)
                else:
                    new_cand[idx] = random.randint(1, 10000)
            
            if len(set(new_cand)) == 12 and all(1 <= x <= 10000 for x in new_cand):
                new_pf = _compute_factor_sets(new_cand)
                new_score = _score_from_pf(new_pf)
                if new_score < current_score:
                    current = new_cand
                    current_pf = new_pf
                    current_score = new_score
    
    # Final verification
    result = current
    assert len(set(result)) == 12
    assert all(1 <= x <= 10000 for x in result)
    
    return result
