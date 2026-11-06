import time
import random
import math

TIME_LIMIT = 85.0

def _init_sieve(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# Precompute smallest prime factor up to 20000
LIMIT = 20000
spf = _init_sieve(LIMIT)

def get_prime_factors(n):
    """Return frozenset of prime factors of n."""
    if n <= 1:
        return frozenset()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return frozenset(factors)

# Precompute prime factors for all numbers up to LIMIT
# This is crucial for speed during local search
pf_cache = [None] * (LIMIT + 1)
for i in range(2, LIMIT + 1):
    pf_cache[i] = get_prime_factors(i)

def score_nums(nums):
    """Count distinct prime factors of all pairwise sums."""
    primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            if s <= LIMIT:
                primes |= pf_cache[s]
            else:
                # Fallback for sums > 20000 (unlikely if B is large enough or nums small)
                primes |= get_prime_factors(s)
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_score = float('inf')
    best_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # Fallback
    
    random.seed(42)
    
    # Candidate bases B
    # We want B to have few prime factors.
    # Multiples of 6 are good (primes 2,3).
    # Multiples of 12 (primes 2,3).
    # Multiples of 30 (primes 2,3,5).
    
    bases = [6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 144, 168, 180, 240, 360, 480, 720, 1200, 2400, 4800]
    
    # Sort bases by number of prime factors?
    # Or just iterate.
    
    for B in bases:
        if time.time() > deadline - 10:
            break
            
        max_u = 10000 // B
        if max_u < 12:
            continue
            
        # Run local search for this base
        # We will perform several restarts
        num_restarts = 5
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize u
            # Strategy 1: Small sequential
            # Strategy 2: Random
            if r == 0:
                u = list(range(1, 13))
            else:
                u = random.sample(range(1, max_u + 1), 12)
            
            current_u = u[:]
            current_score = score_nums([val * B for val in current_u])
            
            if current_score < best_score:
                best_score = current_score
                best_nums = [val * B for val in current_u]
                
            # Simulated Annealing / Hill Climbing
            T = 1.0
            T_min = 0.01
            cooling = 0.995
            
            no_improve = 0
            max_no_improve = 500
            
            while no_improve < max_no_improve and time.time() < deadline - 2:
                idx = random.randint(0, 11)
                old_val = current_u[idx]
                
                # Generate new value
                # Bias towards smaller values?
                # If max_u is large, picking uniformly is bad.
                # Let's pick from a range [1, min(max_u, 1000)] initially?
                # Or just uniform.
                
                # To ensure we explore, let's pick uniformly from [1, max_u]
                new_val = random.randint(1, max_u)
                
                if new_val in current_u:
                    no_improve += 1
                    continue
                
                current_u[idx] = new_val
                new_score = score_nums([val * B for val in current_u])
                
                delta = new_score - current_score
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    no_improve = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_nums = [val * B for val in current_u]
                else:
                    current_u[idx] = old_val
                    no_improve += 1
                
                T *= cooling
                if T < T_min:
                    T = T_min

    # Final check with pure random search if time permits
    if time.time() < deadline - 5:
        for r in range(10):
            if time.time() > deadline - 2:
                break
            nums = random.sample(range(1, 10001), 12)
            current_score = score_nums(nums)
            if current_score < best_score:
                best_score = current_score
                best_nums = nums[:]
                
            # Quick local search
            T = 1.0
            no_improve = 0
            while no_improve < 100 and time.time() < deadline - 2:
                idx = random.randint(0, 11)
                old_val = nums[idx]
                new_val = random.randint(1, 10000)
                if new_val in nums:
                    no_improve += 1
                    continue
                nums[idx] = new_val
                new_score = score_nums(nums)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    no_improve = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_nums = nums[:]
                else:
                    nums[idx] = old_val
                    no_improve += 1
                T *= 0.995
                if T < 0.01: T = 0.01

    return best_nums

if __name__ == "__main__":
    res = construct()
    assert len(res) == 12
    assert len(set(res)) == 12
    assert all(1 <= x <= 10000 for x in res)
    print(res)
    print(score_nums(res))
