"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill
    
    N = 512
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def is_compatible(a, b):
        return lcm(a, b) <= N
    
    def get_divisors(n):
        divs = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)
    
    # Precompute compatibility with 512
    # A number x is compatible with 512 iff lcm(x,512) <= 512
    # 512 = 2^9, so lcm(x, 512) = 512 * x / gcd(x,512)
    # This is <= 512 iff x / gcd(x,512) <= 1 iff x divides 512
    # So only powers of 2 divide 512.
    
    # Strategy: Start with divisors of highly composite numbers, then greedy expand.
    # Key insight: numbers that divide a highly composite number H with H <= 512
    # will have small pairwise lcms with each other.
    
    # Let's think about this differently.
    # We want a set S such that for all a,b in S, lcm(a,b) <= 512.
    # 
    # One strong approach: consider all numbers <= 512 and try to find a large
    # subset where pairwise lcms are bounded.
    #
    # Key structural insight: if we take all divisors of some number M <= 512,
    # then for any a,b dividing M, lcm(a,b) also divides M, so lcm(a,b) <= M <= 512.
    # So the set of divisors of any M <= 512 is valid.
    #
    # But we can do better: we can union divisor sets if the lcms stay bounded.
    #
    # Even better: consider numbers whose prime factorization only involves primes
    # where the exponents are "compatible". 
    #
    # Let's use a greedy approach:
    # 1. Start with a seed (e.g., divisors of 504 or 480 or 512)
    # 2. Try to add each remaining number if it's compatible with all current members
    # 3. Try multiple seeds and keep the best
    
    # Candidate highly composite numbers <= 512:
    # 1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 480, 504, 512
    # Let's check which have the most divisors:
    # 504 = 2^3 * 3^2 * 7 -> (3+1)(2+1)(1+1) = 24 divisors
    # 480 = 2^5 * 3 * 5 -> (5+1)(1+1)(1+1) = 24 divisors
    # 360 = 2^3 * 3^2 * 5 -> 24 divisors
    # 240 = 2^4 * 3 * 5 -> 20 divisors
    # 512 = 2^9 -> 10 divisors
    # 420 = 2^2 * 3 * 5 * 7 -> 24 divisors
    # 336 = 2^4 * 3 * 7 -> 20 divisors
    # 288 = 2^5 * 3^2 -> 18 divisors
    # 252 = 2^2 * 3^2 * 7 -> 18 divisors
    # 168 = 2^3 * 3 * 7 -> 16 divisors
    # 120 = 2^3 * 3 * 5 -> 16 divisors
    # 144 = 2^4 * 3^2 -> 15 divisors
    # 180 = 2^2 * 3^2 * 5 -> 18 divisors
    # 216 = 2^3 * 3^3 -> 16 divisors
    # 240 = 2^4 * 3 * 5 -> 20 divisors
    # 336 = 2^4 * 3 * 7 -> 20 divisors
    # 360 = 2^3 * 3^2 * 5 -> 24 divisors
    # 420 = 2^2 * 3 * 5 * 7 -> 24 divisors
    # 480 = 2^5 * 3 * 5 -> 24 divisors
    # 504 = 2^3 * 3^2 * 7 -> 24 divisors
    
    # But divisor sets alone give at most 24. We can likely do better by mixing.
    
    # Greedy approach with multiple seeds:
    all_numbers = list(range(1, N + 1))
    
    best_set = set()
    
    # Try several seed strategies
    seeds = []
    
    # Seed 1: divisors of 504
    seeds.append(set(get_divisors(504)))
    # Seed 2: divisors of 480
    seeds.append(set(get_divisors(480)))
    # Seed 3: divisors of 360
    seeds.append(set(get_divisors(360)))
    # Seed 4: divisors of 420
    seeds.append(set(get_divisors(420)))
    # Seed 5: divisors of 240
    seeds.append(set(get_divisors(240)))
    # Seed 6: divisors of 512 (powers of 2)
    seeds.append(set(get_divisors(512)))
    # Seed 7: divisors of 336
    seeds.append(set(get_divisors(336)))
    # Seed 8: divisors of 180
    seeds.append(set(get_divisors(180)))
    
    # Also try: all numbers that divide some lcm-bounded structure
    # Numbers <= 512 that are "smooth" in a certain sense
    
    # For each seed, do greedy expansion
    for seed in seeds:
        if time.time() - start_time > DEADLINE:
            break
        
        current = set(seed)
        
        # Sort remaining numbers by some heuristic (e.g., number of divisors, or value)
        remaining = [x for x in all_numbers if x not in current]
        # Try adding in order of increasing value, then by divisibility
        remaining.sort()
        
        for x in remaining:
            if time.time() - start_time > DEADLINE:
                break
            # Check if x is compatible with all elements in current
            compatible = True
            for y in current:
                if not is_compatible(x, y):
                    compatible = False
                    break
            if compatible:
                current.add(x)
        
        if len(current) > len(best_set):
            best_set = set(current)
    
    # Additional greedy: try adding numbers in different orders
    # Sort by number of divisors (descending) - numbers with many divisors are more "central"
    if time.time() - start_time < DEADLINE:
        for seed in seeds:
            current = set(seed)
            remaining = [x for x in all_numbers if x not in current]
            
            # Sort by number of divisors descending
            def num_divisors(n):
                count = 0
                for i in range(1, isqrt(n) + 1):
                    if n % i == 0:
                        count += 1
                        if i != n // i:
                            count += 1
                return count
            
            remaining.sort(key=num_divisors, reverse=True)
            
            for x in remaining:
                if time.time() - start_time > DEADLINE:
                    break
                compatible = True
                for y in current:
                    if not is_compatible(x, y):
                        compatible = False
                        break
                if compatible:
                    current.add(x)
            
            if len(current) > len(best_set):
                best_set = set(current)
    
    # One more pass: try to add numbers sorted by value
    if time.time() - start_time < DEADLINE:
        for seed in seeds:
            current = set(seed)
            remaining = [x for x in all_numbers if x not in current]
            remaining.sort()
            
            for x in remaining:
                if time.time() - start_time > DEADLINE:
                    break
                compatible = True
                for y in current:
                    if not is_compatible(x, y):
                        compatible = False
                        break
                if compatible:
                    current.add(x)
            
            if len(current) > len(best_set):
                best_set = set(current)
    
    if not best_set:
        return [1]
    
    return list(best_set)
# EVOLVE-BLOCK-END
