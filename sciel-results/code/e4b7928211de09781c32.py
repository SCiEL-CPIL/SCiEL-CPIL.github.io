import time
import math

def construct():
    deadline = time.time() + 50
    TARGET = 123457

    # Sieve of Eratosthenes to get primes up to TARGET
    # We need primes up to the upper bound of B, which is at most TARGET
    # But actually, we only need primes up to the current B in the binary search.
    # However, generating primes up to 1000 is very fast.
    # Let's generate primes up to 1000 first. If B needs to be larger, we can extend.
    # Given 123456 has factor 643, B might be around that or lower.
    # Let's just sieve up to 10000 to be safe, it's still fast.
    
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(10000)
    
    def exists_pair_for_bound(B):
        # Get primes <= B
        small_primes = [p for p in primes if p <= B]
        
        # Generate all B-smooth numbers <= TARGET
        smooths = {1}
        for p in small_primes:
            new_smooths = set()
            for s in list(smooths):
                val = s * p
                while val <= TARGET:
                    new_smooths.add(val)
                    val *= p
            smooths.update(new_smooths)
            
        # Check for pair a + b = TARGET
        for a in smooths:
            b = TARGET - a
            if b > 0 and b in smooths:
                return (a, b)
        return None

    # Binary search for minimal B
    # Lower bound: 2
    # Upper bound: We know (1, 123456) works with P=643. So B=643 works.
    # Let's set high to 1000 to be safe, or just 643.
    low = 2
    high = 10000 # Just in case, though 643 is a known upper bound
    
    best_pair = (1, TARGET - 1)
    best_B = 643
    
    while low <= high and time.time() < deadline:
        mid = (low + high) // 2
        pair = exists_pair_for_bound(mid)
        if pair:
            best_pair = pair
            best_B = mid
            high = mid - 1
        else:
            low = mid + 1
            
    return [best_pair[0], best_pair[1]]

# EVOLVE-BLOCK-END
