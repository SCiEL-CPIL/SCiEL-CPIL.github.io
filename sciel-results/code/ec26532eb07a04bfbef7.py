import math
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    # Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
    # Our n is at most (10^9)^2 = 10^18, so this is sufficient.
    # Bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    def is_prime(n):
        if n < 2:
            return False
        # Small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        
        # Write n-1 as d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        while True:
            c = random.randrange(1, n)
            x = random.randrange(2, n)
            y = x
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def has_large_prime_factor(n, limit):
        # Returns True if n has a prime factor > limit
        if n <= limit:
            return False
        
        # Trial division by small primes
        # We only need to check up to sqrt(n) or until n is reduced
        # But we can stop early if we find a factor > limit
        # However, small primes are <= 10^5 usually.
        # Let's just trial divide by primes up to 1000 first to strip small factors.
        # Then use Pollard Rho for the rest.
        
        # Actually, a simpler robust approach for this specific constraint:
        # 1. Trial divide by primes up to 10^4.
        # 2. If remaining n is 1, return False.
        # 3. If remaining n is prime, return n > limit.
        # 4. Else, use Pollard Rho to split, and recurse.
        
        # Pre-computed small primes for trial division
        # Generating primes up to 10^4 is fast
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        # Extend to 1000?
        # Let's generate primes up to 10000 dynamically or hardcode a few.
        # For speed, let's just use a simple loop for small primes.
        
        # Optimization: If n is prime, check directly.
        if is_prime(n):
            return n > limit
            
        # Trial division by small primes (2 to 100)
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                if p > limit:
                    return True
                while n % p == 0:
                    n //= p
                if n == 1:
                    return False
                # Recurse on the reduced n
                if has_large_prime_factor(n, limit):
                    return True
                # If n became 1 or all factors <= limit, continue?
                # Actually, if we stripped p, we still need to check the rest of n.
                # But the recursion above handles the rest of n.
                # Wait, if n % p == 0, we strip all p. Then we check the remaining n.
                # If remaining n has large factor, return True.
                # If not, we are done with this branch?
                # Yes, because we stripped all p.
                # But we might have other small factors.
                # The recursion `has_large_prime_factor(n, limit)` will handle the rest.
                # So we just return the result of the recursion.
                return has_large_prime_factor(n, limit)
        
        # If we are here, n is not divisible by small primes up to 97.
        # And n is not prime (checked at start).
        # So n is composite and has factors > 97.
        # Use Pollard Rho.
        d = pollard_rho(n)
        if d == n:
            # Retry or fallback?
            # Pollard Rho can fail.
            # Let's retry a few times.
            for _ in range(10):
                d = pollard_rho(n)
                if d != n:
                    break
            if d == n:
                # Fallback to trial division up to sqrt(n) (slow but safe)
                # This is rare.
                i = 101
                while i * i <= n:
                    if n % i == 0:
                        if i > limit:
                            return True
                        while n % i == 0:
                            n //= i
                        if n == 1:
                            return False
                        return has_large_prime_factor(n, limit)
                    i += 2
                # n is prime?
                return n > limit

        # Check factors d and n//d
        if has_large_prime_factor(d, limit):
            return True
        if has_large_prime_factor(n // d, limit):
            return True
        return False

    # Sieve of Eratosthenes to generate primes up to N
    def generate_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0:2] = b'\x00\x00'
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = bytearray(len(sieve[i*i:limit+1:i]))
        return [i for i, is_p in enumerate(sieve) if is_p]

    # We search for primes up to 10^7.
    # This is about 664,000 primes.
    # For each prime, we check k numbers.
    # If we find a large k, we update best.
    
    # To maximize performance, we might want to check larger primes first?
    # Larger primes have higher probability of long smooth runs.
    # However, factorization of larger numbers is slightly slower.
    # But the probability gain is significant.
    # Let's check primes in descending order?
    # Or just ascending and hope we find a good one?
    # Ascending is safer to ensure we don't miss small p with weird properties,
    # but for maximizing k, large p is better.
    # Let's do a mix: check a sample of large primes, and all small primes?
    # Given 120s, we can check all primes up to 10^7.
    
    # Let's generate primes up to 10^7.
    # Memory: 10^7 bytes = 10MB. OK.
    # Time: Sieve is fast.
    
    max_p_limit = 10**7
    primes = generate_primes(max_p_limit)
    
    best_k = 0
    best_p = 2
    
    # We can iterate in reverse to find large k faster?
    # If we find k=50 for p=10^7, we can stop or keep going?
    # The task is to maximize k.
    # Let's iterate in reverse to find the best candidate quickly,
    # then if time permits, check smaller ones?
    # Actually, just iterating through all is fine if the check is fast.
    
    # To speed up, we can stop checking a prime p if k reaches 200.
    # Also, if k is small, we break quickly.
    
    # Let's iterate from large to small to find high k values earlier.
    # This allows us to potentially set a high bar and maybe prune?
    # No, we just need the global max.
    # But if we find k=100, we know we have a good result.
    
    for p in reversed(primes):
        # Skip p=2? p=2, p^2+1=5, factor 5>2. k=0.
        # We need k>=1.
        
        # Check sequence p^2 + 1 ... p^2 + k
        k = 0
        p_sq = p * p
        
        # Max k is 200
        for i in range(1, 201):
            n = p_sq + i
            if has_large_prime_factor(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            # Optional: print progress to stderr
            # print(f"New best: p={p}, k={k}", file=sys.stderr)
            
            # If we found k=200, we can't do better.
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
