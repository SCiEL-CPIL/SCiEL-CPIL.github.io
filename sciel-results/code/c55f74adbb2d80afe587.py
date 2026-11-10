"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s limit, leaving margin before 90s kill
    
    MAX_N = 256
    
    # Generate primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # Each prime is initially its own slot
    slots = list(primes)
    
    # Greedily merge slots with best gain while product <= MAX_N
    while True:
        if time.time() > deadline - 5:
            break
        
        best_gain = 0
        best_i = -1
        best_j = -1
        
        n = len(slots)
        for i in range(n):
            for j in range(i + 1, n):
                merged = slots[i] * slots[j]
                if merged > MAX_N:
                    continue
                gain = merged - slots[i] - slots[j]
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
        
        if best_i == -1 or best_gain <= 0:
            break
        
        slots[best_i] = slots[best_i] * slots[best_j]
        slots.pop(best_j)
    
    # Add 1 (coprime with everything)
    result = slots + [1]
    
    # Verify pairwise coprimality (should always hold by construction)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to just primes + 1
                return [1] + primes
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
