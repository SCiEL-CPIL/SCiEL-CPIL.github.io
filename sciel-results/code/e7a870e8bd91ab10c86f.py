import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # 1. Sieve of Eratosthenes for primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]

    # 2. Fast phi function
    def phi(n):
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    best_k = 1
    best_s = 1
    max_n = 10**12

    # Helper to check run length starting at s
    def get_run_length(s):
        if s + 200 > max_n:
            return 0
        seen = set()
        k = 0
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    # Quick scan of small range first
    for s in range(1, 500):
        if time.time() > deadline:
            break
        k = get_run_length(s)
        if k > best_k:
            best_k = k
            best_s = s

    # Random sampling in [1, 10^12]
    random.seed(42)

    while time.time() < deadline:
        # Pick a random start
        s = random.randint(1, max_n - 200)
        
        k = get_run_length(s)
        
        if k > best_k:
            best_k = k
            best_s = s
            
            # Local refinement: try to extend left
            new_s = best_s
            while new_s > 1:
                k_left = get_run_length(new_s - 1)
                if k_left > best_k:
                    best_k = k_left
                    best_s = new_s - 1
                    new_s = best_s
                else:
                    break
            
            # Try extending right by checking s+1
            new_s = best_s
            while new_s + best_k < max_n:
                k_next = get_run_length(new_s + 1)
                if k_next > best_k:
                    best_k = k_next
                    best_s = new_s + 1
                    new_s = best_s
                else:
                    break

    return {'start': best_s, 'length': best_k}
