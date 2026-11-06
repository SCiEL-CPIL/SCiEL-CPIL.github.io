    import time
    import math
    import numpy as np

    def construct():
        start_time = time.time()
        deadline = 115
        
        def is_prime(n):
            if n < 2: return False
            small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            for p in small_primes:
                if n == p: return True
                if n % p == 0: return False
            d = n - 1
            s = 0
            while d % 2 == 0:
                d //= 2
                s += 1
            for a in small_primes:
                if a >= n: continue
                x = pow(a, d, n)
                if x == 1 or x == n - 1: continue
                for _ in range(s - 1):
                    x = (x * x) % n
                    if x == n - 1:
                        break
                else:
                    return False
            return True

        def sieve(limit):
            is_p = bytearray(b'\x01') * (limit + 1)
            is_p[0] = is_p[1] = 0
            for i in range(2, int(limit ** 0.5) + 1):
                if is_p[i]:
                    is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
            return [i for i in range(2, limit + 1) if is_p[i]]

        # Sieve primes up to 5 * 10^6
        LIMIT = 5 * 10**6
        primes = sieve(LIMIT)
        
        best_k = 0
        best_p = 2
        
        # Iterate through primes
        for idx, p in enumerate(primes):
            if time.time() - start_time > deadline:
                break
            
            # Check run length for p
            # Sequential check
            k = 0
            for j in range(1, 201):
                n = p * p + j
                
                # Quick check: if n is prime, it's not p-smooth (since n > p)
                if is_prime(n):
                    break
                
                # Trial division to check p-smoothness
                # We need to divide by all primes q <= p
                # But we can stop if n becomes 1
                # Also, if q > sqrt(n), we can stop and check remainder
                temp_n = n
                is_smooth = False
                
                # We iterate through primes list
                # We only need primes <= p
                # Since primes list is sorted, we can break when q > p
                # But we also break when q*q > temp_n
                
                for q in primes:
                    if q > p:
                        break
                    if q * q > temp_n:
                        break
                    if temp_n % q == 0:
                        while temp_n % q == 0:
                            temp_n //= q
                        if temp_n == 1:
                            is_smooth = True
                            break
                
                if temp_n == 1:
                    is_smooth = True
                else:
                    # temp_n > 1
                    # If we broke because q*q > temp_n, temp_n is prime
                    # If we broke because q > p, temp_n has factor > p (not smooth)
                    # How to distinguish?
                    # If p >= math.isqrt(n):
                    #   We broke because q*q > temp_n (since q <= p implies q <= sqrt(n) is not necessarily true? No, if p >= sqrt(n), then all q <= p are candidates. We stop at q > sqrt(n).)
                    #   So temp_n is prime. Check if temp_n <= p.
                    # Else (p < sqrt(n)):
                    #   We broke because q > p.
                    #   So temp_n has no factor <= p.
                    #   So it has a factor > p. Not smooth.
                    
                    if p >= math.isqrt(n):
                        if is_prime(temp_n) and temp_n <= p:
                            is_smooth = True
                        else:
                            is_smooth = False
                    else:
                        is_smooth = False
                
                if is_smooth:
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
                
        return {'p': best_p, 'k': best_k}
    