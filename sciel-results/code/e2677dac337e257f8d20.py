import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Step 1: Sieve primes up to 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Step 2: Fast phi function with memoization for small values
    phi_cache = {}
    def phi(x):
        if x in phi_cache:
            return phi_cache[x]
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        if x < 10**8:  # Only cache small values to save memory
            phi_cache[x] = result
        return result
    
    # Step 3: Search for long runs of consecutive integers with distinct totients
    best_start = 1
    best_len = 1
    
    random.seed(42)
    
    # Strategy: Generate candidates using structured patterns
    # Key insight: Numbers with "spread" prime factors (different small primes)
    # tend to have more distinct totients in consecutive ranges
    
    candidates = set()
    
    # Pattern 1: Near primorial multiples (numbers with many distinct small prime factors)
    # For a run of k numbers, we want each number to have a unique combination of small primes
    small_primes = primes[:50]  # First 50 primes up to 229
    
    # Generate candidates based on products of subsets of small primes
    # This creates numbers with "diverse" factorizations
    for i in range(min(20, len(small_primes))):
        p = small_primes[i]
        # Try ranges near p * m for various m
        for m in [10**9, 2*10**9, 5*10**9, 10**10, 3*10**10, 7*10**10]:
            base = p * m
            if base + 200 <= 10**12:
                candidates.add(base)
                candidates.add(base - 50)
                candidates.add(base + 50)
    
    # Pattern 2: Structured high ranges with offset patterns
    for base in [10**11, 2*10**11, 3*10**11, 5*10**11, 7*10**11, 9*10**11]:
        for offset in range(0, 10**6, 2000):
            candidates.add(base + offset)
    
    # Pattern 3: Random high ranges (keep some random exploration)
    for _ in range(50):
        s = random.randint(10**10, 10**12 - 200)
        candidates.add(s)
    
    # Sort candidates for deterministic processing
    candidates = sorted(candidates)
    
    # Process candidates with time budget
    for start in candidates:
        if time.time() > deadline:
            break
        
        if start < 1 or start + 200 > 10**12:
            continue
        
        # Quick check: compute phi for up to 200 consecutive numbers
        # and check for distinctness
        seen = set()
        k = 0
        for i in range(200):
            n = start + i
            if n > 10**12:
                break
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
        
        if k > best_len:
            best_len = k
            best_start = start
            
            # If we found a good run, try to extend it in both directions
            if best_len >= 20:
                # Try extending left
                left_start = max(1, best_start - 100)
                seen_left = set()
                for i in range(200, -1, -1):
                    n = best_start + i
                    if n > 10**12:
                        break
                    v = phi(n)
                    seen_left.add(v)
                
                for i in range(best_start - 1, left_start - 1, -1):
                    if time.time() > deadline - 10:
                        break
                    v = phi(i)
                    if v in seen_left:
                        break
                    seen_left.add(v)
                    best_start = i
                    best_len += 1
                
                # Try extending right
                right_end = min(10**12, best_start + best_len + 100)
                seen_right = set()
                for i in range(best_len):
                    n = best_start + i
                    v = phi(n)
                    seen_right.add(v)
                
                for i in range(best_start + best_len, right_end + 1):
                    if time.time() > deadline - 10:
                        break
                    v = phi(i)
                    if v in seen_right:
                        break
                    seen_right.add(v)
                    best_len += 1
    
    # If we have time, do a more thorough search around the best found so far
    remaining_time = deadline - time.time()
    if remaining_time > 30 and best_len > 1:
        # Try to extend or find better runs near the current best
        search_window = min(10**5, int(remaining_time * 500))
        for offset in range(-search_window, search_window + 1, 50):
            new_start = best_start + offset
            if new_start < 1 or new_start + 200 > 10**12:
                continue
            if time.time() > deadline - 5:
                break
            
            seen = set()
            k = 0
            for i in range(200):
                n = new_start + i
                if n > 10**12:
                    break
                v = phi(n)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            
            if k > best_len:
                best_len = k
                best_start = new_start
    
    return {'start': best_start, 'length': best_len}
