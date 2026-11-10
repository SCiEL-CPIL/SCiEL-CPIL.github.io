import math
import time

def construct():
    deadline = time.time() + 140.0
    MAX_N = 10**12
    MAX_K = 200
    
    # 1. Generate primes up to 10^6 for trial division
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Miller-Rabin primality test for large numbers
    def is_prime_mr(n):
        if n < 2:
            return False
        # Small primes check
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Witnesses for n < 3,317,044,064,679,887,385,961,981 (covers up to 10^12 easily)
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Compute phi(n) using primes up to 10^6 and Miller-Rabin for the remainder
    def phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is either prime or a product of two primes > 10^6
            # If temp is prime, we are done.
            # If temp is composite, it must be p*q where p,q > 10^6.
            # Since temp < 10^12, p and q are roughly > 10^6.
            # We check if temp is prime.
            if is_prime_mr(temp):
                result -= result // temp
            else:
                # temp is composite. It must be product of two primes p*q.
                # We need to find p. Since p > 10^6, we can't use the sieve.
                # However, if temp is composite and < 10^12, and we've divided out all primes <= 10^6,
                # then temp is either prime or product of two primes.
                # If product of two primes p*q, we need to find one factor.
                # Fermat's factorization or trial division by primes > 10^6 is slow.
                # But note: if temp is composite, it has a factor < sqrt(temp).
                # sqrt(10^12) = 10^6. We already checked all primes up to 10^6.
                # Wait, if temp is composite, its smallest prime factor must be > 10^6.
                # But then the smallest prime factor p > 10^6 implies p^2 > 10^12, which is impossible for temp <= 10^12.
                # Actually, if p > 10^6, then p^2 > 10^12. So temp cannot have a prime factor > 10^6 if it's composite and <= 10^12?
                # No, temp can be p*q where p=10^6+3, q=10^6+3. Then temp ~ 10^12.
                # But we checked all primes up to 10^6. If p > 10^6, we didn't check it.
                # So we need to factor temp if it's composite.
                # But wait, if we divided out all primes <= 10^6, and temp remains, and temp is composite,
                # its factors must be > 10^6.
                # The product of two numbers > 10^6 is > 10^12.
                # So temp cannot be a product of two primes both > 10^6 if temp <= 10^12.
                # Unless one factor is exactly 10^6? No, 10^6 is not prime.
                # So if temp > 1 and temp <= 10^12 and all prime factors <= 10^6 are removed,
                # then temp must be prime.
                # Proof: Let temp = p1 * p2 * ... * pk.
                # If k >= 2, then p1 >= 10^6 + 1 (since we removed all <= 10^6).
                # Then p1 * p2 >= (10^6+1)^2 > 10^12.
                # But temp <= 10^12. Contradiction.
                # Therefore k=1, so temp is prime.
                # So the `else` branch is never reached for n <= 10^12.
                # But just in case (e.g. if limit was smaller), we handle it.
                # For n <= 10^12 and limit=10^6, this branch is dead code.
                pass
        return result

    # 4. Search for the best run
    # Strategy: Search in a high range where collisions are rare.
    # We want to maximize k. k can be up to 200.
    # If we find a run of 200, we can stop.
    
    # Let's try a few starting points in a high range, e.g., around 10^11.
    # We will check segments of length 200.
    
    best_start = 1
    best_len = 1
    
    # We'll scan a range. How much can we scan?
    # phi computation takes ~100-200 divisions.
    # 200 * 200 = 40,000 phi computations per starting point.
    # We have ~140 seconds.
    # Let's estimate speed. Python is slow. Maybe 10,000 phi/sec?
    # So 40,000 phi takes 4 seconds.
    # We can check maybe 30 starting points.
    
    # Let's pick starting points that are likely to have long runs.
    # Numbers with many small prime factors have smaller phi, but distinctness is about the values.
    # Actually, random numbers in a high range are good.
    
    # Use multiple strategies:
    # 1. Linear scan from a base point
    # 2. Random sampling from various ranges
    # 3. Focus on ranges where numbers are more likely to be prime (higher phi values)
    
    import random
    random.seed(42)  # For reproducibility
    
    # Strategy 1: Linear scan from multiple base points
    base_points = [10**11, 5*10**11, 7*10**11, 9*10**11]
    
    for base in base_points:
        if time.time() > deadline - 10:
            break
            
        # Scan a window of 50 starting points around this base
        for i in range(50):
            if time.time() > deadline - 10:
                break
                
            s = base + i * 10**6
            if s + MAX_K - 1 > MAX_N:
                break
                
            # Compute phi for s ... s+MAX_K-1
            phis = []
            for j in range(MAX_K):
                phis.append(phi(s + j))
                
            # Check for distinctness
            seen = set()
            k = 0
            for val in phis:
                if val in seen:
                    break
                seen.add(val)
                k += 1
                
            if k > best_len:
                best_len = k
                best_start = s
                
            if best_len == MAX_K:
                break
                
        if best_len == MAX_K:
            break
    
    # Strategy 2: Random sampling from various ranges
    if best_len < MAX_K:
        if time.time() > deadline - 10:
            pass
        else:
            # Sample random starting points from various ranges
            ranges = [(10**10, 10**11), (10**11, 5*10**11), (5*10**11, 7*10**11), 
                      (7*10**11, 9*10**11), (9*10**11, 10**12)]
            
            for lo, hi in ranges:
                if time.time() > deadline - 10:
                    break
                    
                # Sample 10 random starting points from this range
                for _ in range(10):
                    if time.time() > deadline - 10:
                        break
                        
                    s = random.randint(lo, hi - MAX_K)
                    
                    # Compute phi for s ... s+MAX_K-1
                    phis = []
                    for j in range(MAX_K):
                        phis.append(phi(s + j))
                        
                    # Check for distinctness
                    seen = set()
                    k = 0
                    for val in phis:
                        if val in seen:
                            break
                        seen.add(val)
                        k += 1
                        
                    if k > best_len:
                        best_len = k
                        best_start = s
                        
                    if best_len == MAX_K:
                        break
                
                if best_len == MAX_K:
                    break
    
    return {'start': best_start, 'length': best_len}
