import time
import random
import math
from itertools import combinations

def _factorize(n, small_primes):
    """Return set of prime factors of n using given small primes, then trial division."""
    factors = set()
    temp = n
    for p in small_primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factors.add(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.add(temp)
    return factors

def _count_distinct_primes(arr, small_primes):
    """Count distinct primes across all pairwise sums."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            pf = _factorize(s, small_primes)
            all_primes.update(pf)
    return len(all_primes)

def _small_primes_up_to(limit):
    """Generate primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    return [i for i in range(2, limit+1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute small primes for fast factorization
    small_primes = _small_primes_up_to(20000)
    
    best_arr = list(range(1, 13))
    best_score = _count_distinct_primes(best_arr, small_primes)
    
    # Strategy: try to find sets where all pair sums use few primes
    # Key insight: if all numbers are congruent to r mod p for odd prime p,
    # then all pair sums are 2r mod p. If 2r != 0 mod p, p never divides any sum.
    
    # Try structured constructions first
    # All odd, all ≡ 1 mod 3, all ≡ 1 mod 5: sums are even, not div by 3 or 5
    # Numbers: 1, 16, 31, 46, 61, 76, 91, 106, 121, 136, 151, 166 (all ≡1 mod 15, odd)
    # Check: 1 mod 15: 1,16,31,46,61,76,91,106,121,136,151,166 - all odd? 1(odd),16(even)... no
    # Need odd AND ≡1 mod 15: 1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301, 331
    # All odd, all ≡ 1 mod 3, all ≡ 1 mod 5. Sums ≡ 2 mod 3, ≡ 2 mod 5, even.
    structured1 = [1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301, 331]
    s1 = _count_distinct_primes(structured1, small_primes)
    if s1 < best_score:
        best_score = s1
        best_arr = structured1[:]
    
    # Try all ≡ 2 mod 15, odd: 2 is even. Try ≡ 8 mod 15: 8 even. 
    # ≡ 14 mod 15: 14 even. ≡ 4 mod 15: 4 even. ≡ 10 mod 15: 10 even.
    # For odd numbers ≡ r mod 15: r must be odd: 1, 3, 5, 7, 9, 11, 13
    # r=1: sums≡2 mod3, ≡2 mod5 (good, avoids 3,5)
    # r=7: 7 mod 3 = 1, 7 mod 5 = 2. sums: 14 mod 3 = 2, 14 mod 5 = 4. Avoids 3,5.
    # r=13: 13 mod 3 = 1, 13 mod 5 = 3. sums: 26 mod 3 = 2, 26 mod 5 = 1. Avoids 3,5.
    
    # Try r=7 mod 15, odd: 7, 37, 67, 97, 127, 157, 187, 217, 247, 277, 307, 337
    structured2 = [7, 37, 67, 97, 127, 157, 187, 217, 247, 277, 307, 337]
    s2 = _count_distinct_primes(structured2, small_primes)
    if s2 < best_score:
        best_score = s2
        best_arr = structured2[:]
    
    # Random local search
    random.seed(42)
    
    def _evaluate(arr):
        return _count_distinct_primes(arr, small_primes)
    
    # Try random starts and local improvement
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Random start: pick 12 random odd numbers
        arr = random.sample(range(1, 10001, 2), 12)
        arr.sort()
        score = _evaluate(arr)
        
        if score < best_score:
            best_score = score
            best_arr = arr[:]
        
        # Local search: try swapping one element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                for _ in range(20):
                    new_val = random.randint(1, 10000)
                    if new_val in arr:
                        continue
                    old_val = arr[i]
                    arr[i] = new_val
                    arr.sort()
                    new_score = _evaluate(arr)
                    if new_score < score:
                        score = new_score
                        improved = True
                        if score < best_score:
                            best_score = score
                            best_arr = arr[:]
                        break
                    else:
                        arr[i] = old_val
                        arr.sort()
        
        if best_score <= 3:
            break
    
    return best_arr
