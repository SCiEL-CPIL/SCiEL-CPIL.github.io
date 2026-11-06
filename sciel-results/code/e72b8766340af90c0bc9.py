"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 142.0  # safety margin before 150s kill
    
    def now():
        return time.time() - start_time
    
    # Precompute primes up to 10^6 for trial division
    limit = 1000000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    def phi(x):
        """Compute Euler's totient function phi(x) with exact factorization."""
        if x <= 1:
            return 1 if x == 1 else 0
        result = x
        y = x
        for p in small_primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result
    
    # Phase 1: Exhaustive search in small range [1, 50000]
    N1 = 50000
    values = [0] + [phi(x) for x in range(1, N1 + 1)]
    
    best_k = 1
    best_s = 1
    
    for s in range(1, N1 - 199):
        seen = set()
        k = 0
        while s + k <= N1 and values[s + k] not in seen:
            seen.add(values[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
        if now() > DEADLINE - 60:
            break
    
    # Phase 2: Random sampling in larger ranges
    # Try random starting points in [10^6, 10^12]
    attempts = 0
    max_attempts = 5000
    
    while now() < DEADLINE - 5 and attempts < max_attempts:
        attempts += 1
        
        # Choose a random starting point
        # Bias towards smaller values where runs might be longer
        if random.random() < 0.3:
            s = random.randint(1000, 1000000)
        elif random.random() < 0.5:
            s = random.randint(1000000, 100000000)
        else:
            s = random.randint(100000000, 1000000000000)
        
        if s + 199 > 10**12:
            s = 10**12 - 199
        
        # Compute phi values for a window starting at s
        seen = set()
        k = 0
        max_check = min(200, 10**12 - s + 1)
        
        for i in range(max_check):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            # If we found a good run, try nearby starts
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
                if now() > DEADLINE - 5:
                    break
                s2 = s + delta
                if s2 < 1 or s2 + 199 > 10**12:
                    continue
                seen2 = set()
                k2 = 0
                max_check2 = min(200, 10**12 - s2 + 1)
                for i in range(max_check2):
                    val = phi(s2 + i)
                    if val in seen2:
                        break
                    seen2.add(val)
                    k2 += 1
                if k2 > best_k:
                    best_k = k2
                    best_s = s2
    
    # Phase 3: Local optimization around best found start
    if best_k > 1 and now() < DEADLINE - 10:
        for delta in range(-20, 21):
            if now() > DEADLINE - 5:
                break
            s2 = best_s + delta
            if s2 < 1 or s2 + 199 > 10**12:
                continue
            seen = set()
            k = 0
            max_check = min(200, 10**12 - s2 + 1)
            for i in range(max_check):
                val = phi(s2 + i)
                if val in seen:
                    break
                seen.add(val)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s2
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
