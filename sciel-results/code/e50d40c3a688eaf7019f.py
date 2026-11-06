import time
import random
import math

def construct():
    # Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # Fast phi function
    def phi(n):
        res = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                res -= res // p
        if temp > 1:
            res -= res // temp
        return res

    deadline = time.time() + 105  # Safety margin
    best_k = 1
    best_s = 1
    max_range = 10**12
    
    # Helper to check and update best
    def check_run(start):
        nonlocal best_k, best_s
        seen = set()
        k = 0
        # We compute phi for start to start+199
        # But we stop early if collision
        for i in range(200):
            val = phi(start + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = start
            
            # Try to extend
            # Extend left
            # We need to check if phi(start-1) is not in seen
            # And phi(start-2) etc.
            # But we don't have the values for left side.
            # We can re-compute or store them.
            # For simplicity, just check a few steps left.
            # Actually, extending is complex because we need the set of values.
            # Let's just rely on random sampling to hit the start of the run.
            
            if best_k == 200:
                return True # Max reached
        
        return False

    # Random search
    while time.time() < deadline:
        s = random.randint(1, max_range)
        # Check if we have time for a full check
        if time.time() > deadline - 2:
            break
            
        # Run check
        check_run(s)
        
        # If we found a very long run, we can stop
        if best_k == 200:
            break

    return {'start': best_s, 'length': best_k}
