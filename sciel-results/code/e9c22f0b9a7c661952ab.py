import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    # 1. Generate primes up to 10^7 for fast phi computation
    # We need primes up to sqrt(10^12) = 10^6, but having more helps for trial division
    # Actually, for phi(n) where n can be up to 10^12, we need primes up to 10^6 for trial division.
    # If n has a factor > 10^6, the remaining part is prime (since n <= 10^12, it can have at most one such factor).
    
    limit_p = 10**6
    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def phi(x):
        if x <= 1:
            return 1
        result = x
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        return result

    # 2. Search for long runs
    # Strategy: Random sampling of large numbers.
    # Collisions are rare. We want to find a run of length 200.
    # The maximum possible length is 200.
    # If we find a run of length 200, we can return it immediately (maximized k).
    # If not, we return the best found.
    
    best_start = 1
    best_len = 1
    
    # Helper to check a run starting at s
    def check_run(s, target_len=200):
        seen = set()
        k = 0
        for i in range(target_len):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    # First, try some small ranges just in case (unlikely to be best, but cheap)
    # The seed program found best in 1..20000. Let's assume that's small.
    
    # Random search in large ranges
    # We sample random starting points s in [1, 10^12 - 200]
    # We check if phi(s)..phi(s+199) are distinct.
    
    # To be efficient, we can use a quick check first.
    # But phi is expensive.
    # Let's just do random sampling.
    # How many samples?
    # Each check takes ~200 * cost(phi).
    # cost(phi) for n~10^12 is ~number of primes up to 10^6 that divide n? No, trial division up to sqrt(n).
    # Worst case n is prime, we check all primes up to 10^6 (~78k primes).
    # 200 * 78k = 15.6M operations per sample.
    # In Python, 15M ops ~ 0.1-0.5s? Maybe more.
    # So maybe 100-500 samples in 110s.
    
    # Let's optimize phi slightly.
    # Precompute primes up to 10^6.
    
    # We can also try to be smarter.
    # But random sampling worked for the "island 1" improvement (score 2.77).
    # Score 2.77 implies k=200?
    # Seed score 1.0 implies k=?
    # Seed program:
    # values = [0] + [phi(x) for x in range(1, 20001)]
    # best = max(k, s)
    # If best k was ~72 (since 2.77 / 1.0 = 2.77, and 200/72 = 2.77), then yes.
    # 200 / 72 = 2.777...
    # So the seed likely found a run of length 72.
    # The improvement found a run of length 200.
    
    # So we just need to find a run of length 200.
    # Random sampling should work.
    
    # Let's add a deterministic fallback if random fails?
    # Or just random.
    
    # We will try random starts.
    # To avoid duplicates, we don't need to track them strictly, just sample.
    
    # Let's use a few fixed "good" seeds if we know any? No, we don't.
    # Just random.
    
    # Optimization:
    # Check time every few samples.
    
    max_samples = 1000 # Cap
    attempts = 0
    
    while time.time() - start_time < DEADLINE and attempts < max_samples:
        # Pick a random start
        # s + 199 <= 10^12
        s = random.randint(1, 10**12 - 200)
        
        # Quick check: are there any obvious collisions?
        # Maybe check phi(s) and phi(s+1) first?
        # No, just do the full check.
        
        k = check_run(s, 200)
        
        if k > best_len:
            best_len = k
            best_start = s
            if best_len == 200:
                break # Found maximum possible
        
        attempts += 1
        
        # Time check
        if attempts % 10 == 0:
            if time.time() - start_time > DEADLINE:
                break

    # If we didn't find 200, maybe we can do a local search around the best found?
    # Or just return best.
    
    # One more thing: The problem says "Long runs below 10^12 are finite evidence".
    # It implies we should maximize k.
    # If random doesn't find 200, we return best found.
    
    return {'start': best_start, 'length': best_len}
