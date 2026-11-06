import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 115  # seconds, safety margin

    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    # Witnesses: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    def is_prime(n):
        if n < 2:
            return False
        # Small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as d * 2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Test witnesses
        for a in small_primes:
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

    # Pollard's Rho with Brent's cycle detection
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128  # batch size
            g = r = q = 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g != 1:
                        break
            if g != n:
                return g

    # Precompute small primes for trial division
    # We need primes up to some limit. Let's say 10^6.
    # This helps in stripping small factors quickly.
    def sieve(limit):
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = 0
        return [i for i in range(limit + 1) if is_p[i]]

    small_primes = sieve(10**6)
    # Create a set for O(1) lookup? No, we need to iterate.
    # But for trial division in is_p_smooth, we just iterate.

    def is_p_smooth(n, p):
        if n <= p:
            return True
        if is_prime(n):
            return n <= p
        
        # Trial division by small primes
        # We only need to divide by primes <= p
        # But our list is up to 10^6.
        # If p > 10^6, we only divide by primes in our list.
        # If p < 10^6, we divide by primes in our list up to p.
        
        for q in small_primes:
            if q > p:
                break
            if n % q == 0:
                # Divide out all factors of q
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                if n <= p:
                    return True
                # If n is prime here, and n > p, it fails.
                # But we continue to check if n is composite.
        
        if n == 1:
            return True
        if n <= p:
            return True
        
        # If n is prime here, it must be > p (since we divided out all small factors)
        if is_prime(n):
            return False
        
        # Find a factor using Pollard Rho
        d = pollard_rho(n)
        if d == n:
            # Should not happen if is_prime is correct
            return False
        
        # Recurse
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    best_k = 0
    best_p = 2
    
    # Search primes in increasing order
    # We can generate primes on the fly or sieve a large block.
    # Let's sieve up to 10^7 first.
    
    def search_primes(start, end):
        # Simple sieve for range [start, end]
        # Or just use the small_primes list if end <= 10^6
        if end <= 10**6:
            return [p for p in small_primes if start <= p <= end]
        
        # For larger ranges, we need a segmented sieve or just trial division by small primes.
        # Given time constraints, let's just iterate and check primality.
        # This is slower but sufficient for 10^7 range if we are careful.
        primes = []
        for i in range(start, end + 1):
            if i % 2 == 0 and i != 2:
                continue
            if is_prime(i):
                primes.append(i)
        return primes

    # Initial search with small primes
    for p in small_primes:
        if time.time() - start_time > deadline - 10:
            break
            
        k = 0
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p

    # Search larger primes
    # We can jump to larger ranges.
    # Let's try up to 10^8 if time permits.
    # But checking primality for each number up to 10^8 is slow.
    # Let's use a segmented sieve or just check a sparse set.
    # Actually, the probability of finding a large k is low.
    # The best k found so far might be small.
    # Let's try to search primes up to 10^7 using a faster method.
    
    # Re-sieve up to 10^7?
    # Sieve(10^7) takes ~1s.
    if time.time() - start_time < deadline - 20:
        primes_10m = sieve(10**7)
        # Skip those already checked
        prev_limit = 10**6
        for p in primes_10m:
            if p <= prev_limit:
                continue
            if time.time() - start_time > deadline - 5:
                break
            
            k = 0
            while k < 200:
                n = p * p + k + 1
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p

    # If we have time, search even larger?
    # Maybe not, as primality testing becomes the bottleneck.
    # But let's try a few random large primes?
    # No, structured search is better.
    
    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
