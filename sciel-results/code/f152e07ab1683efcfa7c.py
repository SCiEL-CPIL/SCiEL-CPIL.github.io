"""Rewritten program: random sampling of large ranges to find long runs of distinct totients."""
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135.0  # Safety margin before 150s kill
    
    # Precompute primes up to 10^6 for fast factorization
    # sqrt(10^12) = 10^6
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, limit_p + 1) if is_prime[i]]
    
    def phi_fast(n):
        """Compute Euler's totient using precomputed primes."""
        result = n
        y = n
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def check_range(start, length):
        """Check if phi values for [start, start+length-1] are all distinct."""
        seen = set()
        for i in range(length):
            val = phi_fast(start + i)
            if val in seen:
                return False
            seen.add(val)
        return True
    
    def find_max_run_from(start, max_len=200):
        """Find the maximum run length starting at 'start' up to max_len."""
        seen = set()
        k = 0
        while k < max_len:
            val = phi_fast(start + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    best_k = 1
    best_start = 1
    
    # Strategy: Randomly sample starting points in [1, 10^12] and check for long runs
    # For large numbers, totient collisions are rare, so long runs should be common
    # We want to find a run of length 200 (the maximum allowed)
    
    # First, try some structured approaches
    # Try starting near large powers of 2 or primes
    
    # Random search
    random.seed(42)
    
    # Try to find a run of length 200
    # Sample random starting points
    attempts = 0
    while time.time() < DEADLINE and best_k < 200:
        attempts += 1
        
        # Generate random start points
        # Bias towards larger numbers where collisions are rarer
        # But we need start + 199 <= 10^12, so start <= 10^12 - 199
        
        # Try different ranges
        if attempts % 10 == 0:
            # Occasionally try structured starts
            for base in [10**6, 10**7, 10**8, 10**9, 10**10, 10**11]:
                if base + 200 > 10**12:
                    break
                start = base
                k = find_max_run_from(start, 200)
                if k > best_k:
                    best_k = k
                    best_start = start
                    if best_k >= 200:
                        break
            if best_k >= 200:
                break
        
        # Random sampling
        # Use logarithmic distribution to cover many orders of magnitude
        log_start = random.uniform(0, math.log10(10**12 - 200))
        start = int(10**log_start)
        if start < 1:
            start = 1
        if start + 200 > 10**12:
            start = 10**12 - 200
        
        k = find_max_run_from(start, 200)
        if k > best_k:
            best_k = k
            best_start = start
            if best_k >= 200:
                break
        
        # Also try nearby starts if we found a decent run
        if k >= 50 and k < 200:
            # Search around this start for a better one
            for delta in [-5, -3, -1, 1, 3, 5]:
                s2 = start + delta
                if s2 >= 1 and s2 + 200 <= 10**12:
                    k2 = find_max_run_from(s2, 200)
                    if k2 > best_k:
                        best_k = k2
                        best_start = s2
                        if best_k >= 200:
                            break
            if best_k >= 200:
                break
    
    # If we haven't found 200, do a more focused search
    if best_k < 200 and time.time() < DEADLINE:
        # Try systematic sampling in high ranges
        for base in range(10**11, 10**12 - 200, 10**7):
            if time.time() >= DEADLINE:
                break
            start = base
            k = find_max_run_from(start, 200)
            if k > best_k:
                best_k = k
                best_start = start
                if best_k >= 200:
                    break
            # Try a few random offsets
            for _ in range(3):
                if time.time() >= DEADLINE:
                    break
                offset = random.randint(0, 10**7)
                start = base + offset
                if start + 200 > 10**12:
                    continue
                k = find_max_run_from(start, 200)
                if k > best_k:
                    best_k = k
                    best_start = start
                    if best_k >= 200:
                        break
            if best_k >= 200:
                break
    
    return {'start': best_start, 'length': best_k}
