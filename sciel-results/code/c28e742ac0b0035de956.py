"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import isqrt

# Precompute Smallest Prime Factors (SPF) up to 20000
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, isqrt(MAX_SUM) + 1):
    if spf[i] == i:
        for j in range(i * i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

def _get_prime_factors(n):
    """Return set of prime factors of n using precomputed SPF."""
    factors = set()
    while n > 1:
        factors.add(spf[n])
        n //= spf[n]
    return factors

def construct():
    """
    Find 12 distinct positive integers <= 10000 minimizing the number of distinct primes
    dividing sums of distinct-index pairs.
    """
    best_nums = None
    best_cost = float('inf')
    
    # Number of restarts for local search
    restarts = 200
    max_iter_per_restart = 1000
    
    for _ in range(restarts):
        # Initialize with a random set of 12 distinct numbers
        if random.random() < 0.5:
            current_nums = random.sample(range(1, 10001), 12)
        else:
            # Structured: multiples of a random power of 2
            pow2 = 2 ** random.randint(1, 4)
            candidates = [k * pow2 for k in range(1, 10000 // pow2 + 1)]
            if len(candidates) >= 12:
                current_nums = random.sample(candidates, 12)
            else:
                current_nums = random.sample(range(1, 10001), 12)
                
        # Initialize prime counts
        prime_counts = {}
        
        def add_prime_factors(n, delta):
            factors = _get_prime_factors(n)
            for p in factors:
                if p not in prime_counts:
                    prime_counts[p] = 0
                prime_counts[p] += delta
                if prime_counts[p] == 0:
                    del prime_counts[p]
        
        # Build initial state
        for i in range(12):
            for j in range(i + 1, 12):
                add_prime_factors(current_nums[i] + current_nums[j], 1)
                
        current_cost = len(prime_counts)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_nums = current_nums[:]
            
        # Local Search
        for _ in range(max_iter_per_restart):
            idx = random.randint(0, 11)
            old_val = current_nums[idx]
            
            new_val = random.randint(1, 10000)
            if new_val == old_val:
                continue
            if new_val in current_nums:
                continue
                
            # Remove old sums involving idx
            for j in range(12):
                if j != idx:
                    add_prime_factors(old_val + current_nums[j], -1)
                    
            # Tentatively set new value
            current_nums[idx] = new_val
            
            # Add new sums involving idx
            for j in range(12):
                if j != idx:
                    add_prime_factors(new_val + current_nums[j], 1)
                    
            new_cost = len(prime_counts)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_nums = current_nums[:]
            else:
                # Revert
                for j in range(12):
                    if j != idx:
                        add_prime_factors(new_val + current_nums[j], -1)
                
                current_nums[idx] = old_val
                
                for j in range(12):
                    if j != idx:
                        add_prime_factors(old_val + current_nums[j], 1)
                        
            if best_cost <= 2:
                return best_nums

    return best_nums
# EVOLVE-BLOCK-END
