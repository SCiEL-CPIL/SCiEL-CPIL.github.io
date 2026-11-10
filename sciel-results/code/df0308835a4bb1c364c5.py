"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 80.0

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

def _prime_factors_fast_tuple(n):
    """Return tuple of prime factors of n (for faster set operations)."""
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

def _build_prime_set(nums):
    """Return the set of all distinct primes dividing any pair sum."""
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            all_primes |= _prime_factors_fast(nums[i] + nums[j])
    return all_primes

def construct():
    random.seed(42)
    
    # Precompute smooth numbers (2^a * 3^b * 5^c * 7^d) up to 20000 ONCE
    smooth_numbers = []
    for a in range(15):
        p2 = 2**a
        if p2 > 20000:
            break
        for b in range(12):
            p3 = 3**b
            if p2 * p3 > 20000:
                break
            for c in range(9):
                p5 = 5**c
                if p2 * p3 * p5 > 20000:
                    break
                for d in range(7):
                    p7 = 7**d
                    smooth = p2 * p3 * p5 * p7
                    if smooth > 20000:
                        break
                    smooth_numbers.append(smooth)
    smooth_numbers = sorted(set(smooth_numbers))
    
    # Strategy: Use a small set of primes and try to make all pair sums
    # have prime factors only from that set.
    
    # First, generate several initial candidates using different strategies
    candidates = []
    
    # Strategy 1: Powers of 2 (sums of two powers of 2 have limited prime factors)
    candidates.append([2**i for i in range(1, 13)])
    
    # Strategy 2: All multiples of a highly composite number
    candidates.append([360 * (i+1) for i in range(12)])
    
    # Strategy 3: Small numbers that are multiples of 30
    candidates.append([30 * (i+1) for i in range(12)])
    
    # Strategy 4: Numbers of form k*12 + 6 (all odd multiples of 6)
    candidates.append([6 * (2*i+1) for i in range(12)])
    
    # Strategy 5: Consecutive even numbers
    candidates.append(list(range(2, 25, 2)))
    
    # Strategy 6: Numbers that are all 1 mod small primes
    candidates.append([60 * i + 1 for i in range(1, 13)])
    
    # Strategy 7: Use Chinese remainder to make sums divisible by 2,3,5,7
    candidates.append([210 * (i+1) for i in range(12)])
    
    # Strategy 8: Use numbers that are all close to each other (small range)
    candidates.append([1000 + i for i in range(12)])
    
    # Strategy 9: Use numbers that are all powers of 2 plus a constant
    candidates.append([1 + 2**i for i in range(1, 13)])
    
    # Strategy 10: Fibonacci-like
    fibs = [1, 2]
    for _ in range(10):
        fibs.append(fibs[-1] + fibs[-2])
    candidates.append(fibs)
    
    # Strategy 11: All numbers equal to a highly composite number times small factors
    candidates.append([27720 // (i+1) for i in range(12) if 27720 % (i+1) == 0][:12])
    if len(candidates[-1]) < 12:
        candidates.pop()
    
    # Strategy 12: Numbers of form 2^a * 3^b for various (a,b)
    candidates.append([])
    for a in range(5):
        for b in range(5):
            val = 2**a * 3**b
            if 1 <= val <= 10000 and val not in candidates[-1]:
                candidates[-1].append(val)
            if len(candidates[-1]) >= 12:
                break
        if len(candidates[-1]) >= 12:
            break
    if len(candidates[-1]) < 12:
        candidates.pop()
    
    # Strategy 13: CRT-based - make all sums divisible by 2,3,5,7
    candidates.append([210 * (i+1) for i in range(12)])
    
    # Strategy 14: Make sums divisible by 2,3,5 using CRT with offset
    candidates.append([30 * (i+1) for i in range(12)])
    
    # Strategy 16: Arithmetic progression with common difference = lcm of small primes
    for a in [1, 2, 3, 5, 7, 10, 14, 15, 20, 21, 30, 35, 42, 70, 105, 210]:
        cand = [a + 210 * i for i in range(12)]
        if all(1 <= x <= 10000 for x in cand) and len(set(cand)) == 12:
            candidates.append(cand)
    
    # Strategy 17: Use powers of 2 times a base
    for base in [6, 12, 30, 60, 120, 210, 420]:
        cand = []
        for e in range(15):
            val = base * (2**e)
            if 1 <= val <= 10000 and val not in cand:
                cand.append(val)
            if len(cand) >= 12:
                break
        if len(cand) == 12:
            candidates.append(cand)
    
    # Strategy 18: Greedy conflict-graph approach - pick numbers whose pairwise sums
    # introduce minimal new primes. Start with a small prime set S and greedily
    # select 12 numbers such that all pair sums have factors only in S (or close).
    # For each small prime set S, try to find 12 numbers where pair sums are S-smooth.
    
    # Evaluate all candidates
    best_candidate = None
    best_score = float('inf')
    best_pf = None
    
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
            best_pf = pf
    
    if best_candidate is None:
        best_candidate = list(range(1, 13))
        best_pf = _compute_factor_sets(best_candidate)
        best_score = _total_factor_count(best_pf)
    
    # Local search: try to reduce the number of distinct primes
    current = best_candidate[:]
    current_pf = best_pf
    current_score = best_score
    
    n = 12
    
    # Precompute current prime set
    current_primes = set()
    for i in range(n):
        for j in range(i+1, n):
            current_primes |= current_pf[i][j]
    
    # Hill climbing: try changing each element to reduce prime count
    # Precompute base primes (pairs not involving a given index)
    def _base_primes_excluding(pf, exclude_idx):
        """Get union of all pair factors excluding pairs involving exclude_idx."""
        base = set()
        for i in range(n):
            if i == exclude_idx:
                continue
            for j in range(i+1, n):
                if j == exclude_idx:
                    continue
                base |= pf[i][j]
        return base
    
    # Precompute base primes for each index
    base_primes_cache = [_base_primes_excluding(current_pf, i) for i in range(n)]
    
    # Precompute candidate value pools for faster local search
    # Values that are multiples of small primes (likely to produce smooth sums)
    good_candidates = set()
    for mult in range(1, 10001):
        # Check if mult is divisible by at least one of 2,3,5,7
        if mult % 2 == 0 or mult % 3 == 0 or mult % 5 == 0 or mult % 7 == 0:
            good_candidates.add(mult)
    good_candidates = list(good_candidates)
    random.shuffle(good_candidates)
    
    for restart in range(300):
        if _deadline_hit():
            break
        
        improved = True
        while improved and not _deadline_hit():
            improved = False
            
            for idx in range(n):
                if _deadline_hit():
                    break
                
                old_val = current[idx]
                
                # Precompute base primes for this index
                base_primes = base_primes_cache[idx]
                
                best_new_val = None
                best_new_score = current_score
                best_new_pf = None
                best_new_primes = None
                
                # Generate candidate values efficiently
                cand_vals = set()
                
                # Small range around old value
                for d in range(-30, 31):
                    v = old_val + d
                    if 1 <= v <= 10000:
                        cand_vals.add(v)
                
                # For each other element, add smooth - current[j] as candidate
                # Use precomputed smooth_numbers (generated once above)
                for j in range(n):
                    if j == idx:
                        continue
                    for smooth in smooth_numbers[:300]:
                        v = smooth - current[j]
                        if 1 <= v <= 10000:
                            cand_vals.add(v)
                
                # Also try values that make sums divisible by small primes
                for target_prime in [2, 3, 5, 7, 11, 13]:
                    for j in range(n):
                        if j == idx:
                            continue
                        rem = (-current[j]) % target_prime
                        # Sample a few values from this residue class
                        for _ in range(5):
                            k = random.randint(0, 9999) * target_prime
                            v = k + rem
                            if 1 <= v <= 10000:
                                cand_vals.add(v)
                
                # Add some random good candidates
                for v in good_candidates[:50]:
                    cand_vals.add(v)
                
                # Also try random values
                for _ in range(20):
                    cand_vals.add(random.randint(1, 10000))
                
                # Remove values already in current (to keep distinctness)
                current_set = set(current)
                cand_vals -= current_set
                
                # Limit candidates to keep computation fast
                if len(cand_vals) > 600:
                    # Randomly sample
                    cand_vals = set(random.sample(list(cand_vals), 600))
                
                # Evaluate candidates
                for new_val in cand_vals:
                    if new_val in current:
                        continue
                    
                    # Compute new prime set using precomputed base
                    new_primes = set(base_primes)
                    
                    # Pairs involving idx
                    for j in range(n):
                        if j == idx:
                            continue
                        s = new_val + current[j]
                        pf = _prime_factors_fast(s)
                        new_primes |= pf
                    
                    new_score = len(new_primes)
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                        best_new_primes = new_primes
                        # Save the new pf
                        best_new_pf = [row[:] for row in current_pf]
                        for j in range(n):
                            if j != idx:
                                s = new_val + current[j]
                                pf = _prime_factors_fast(s)
                                best_new_pf[idx][j] = pf
                                best_new_pf[j][idx] = pf
                
                if best_new_val is not None and best_new_score < current_score:
                    current[idx] = best_new_val
                    current_pf = best_new_pf
                    current_score = best_new_score
                    current_primes = best_new_primes
                    # Update base primes cache
                    base_primes_cache = [_base_primes_excluding(current_pf, i) for i in range(n)]
                    improved = True
                    break
            
            # If no improvement, try a random perturbation
            if not improved and not _deadline_hit():
                # Try changing 1-2 elements randomly
                for _ in range(3):
                    if _deadline_hit():
                        break
                    new_cand = current[:]
                    num_changes = random.randint(1, 2)
                    for _ in range(num_changes):
                        idx = random.randint(0, 11)
                        new_cand[idx] = random.choice(good_candidates[:200])
                    
                    if len(set(new_cand)) == 12 and all(1 <= x <= 10000 for x in new_cand):
                        new_pf = _compute_factor_sets(new_cand)
                        new_score = _total_factor_count(new_pf)
                        if new_score < current_score:
                            current = new_cand
                            current_pf = new_pf
                            current_score = new_score
                            base_primes_cache = [_base_primes_excluding(current_pf, i) for i in range(n)]
                            break
    
    # Final verification
    result = current
    if len(set(result)) != 12 or not all(1 <= x <= 10000 for x in result):
        # Fallback to a valid solution
        result = [360 * (i+1) for i in range(12)]
    
    return result
