import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Generate primes up to a limit
    # We want to search p up to ~5e6 or 1e7
    # Let's try 5e6 first to be safe on time
    limit = 5_000_000
    
    # Sieve of Eratosthenes
    is_prime_arr = bytearray([1]) * (limit + 1)
    is_prime_arr[0] = 0
    is_prime_arr[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime_arr[i]:
            is_prime_arr[i*i : limit+1 : i] = bytearray(len(range(i*i, limit+1, i)))
    
    primes = [i for i, v in enumerate(is_prime_arr) if v]
    
    # Miller-Rabin Primality Test
    def is_prime_mr(n):
        if n < 2:
            return False
        # Small primes check
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        
        # Write n-1 as d * 2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        # Bases for deterministic MR for n < 2^64
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
        for a in bases:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def is_p_smooth(n, p, primes_list):
        if n == 1:
            return True
        # If n is prime, it is > p (since n >= p^2+1 > p for p>=2)
        # So it is not p-smooth
        if is_prime_mr(n):
            return False
            
        rem = n
        for q in primes_list:
            if q > p:
                break
            if q * q > rem:
                break
            while rem % q == 0:
                rem //= q
        
        if rem == 1:
            return True
        
        # If we broke because q > p, rem has a factor > p
        # If we broke because q*q > rem, rem is prime
        # Check if rem <= p
        if rem <= p:
            return True
        else:
            return False

    best_k = 0
    best_p = 2
    
    # Iterate through primes
    # We can skip very small primes if we want, but they are fast
    for p in primes:
        if time.time() > deadline:
            break
            
        # Optional: skip if p is too large? 
        # The probability is roughly constant, so larger p doesn't help much 
        # but allows larger k? No, k is bounded by 200.
        # Actually, for very large p, p^2+i is less likely to be smooth?
        # No, u=2 is constant.
        
        i = 1
        while i <= 200:
            n = p * p + i
            if is_p_smooth(n, p, primes):
                i += 1
            else:
                break
        
        if i - 1 > best_k:
            best_k = i - 1
            best_p = p
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
