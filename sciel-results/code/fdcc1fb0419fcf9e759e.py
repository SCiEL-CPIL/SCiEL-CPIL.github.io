"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # The problem is to find the largest subset of {2..40} with distinct subset products.
    # This is equivalent to finding the maximum number of multiplicatively independent integers.
    # The number of primes in {2..40} is 12.
    # The maximum size of a multiplicatively independent set is bounded by the number of primes, which is 12.
    # The set of primes itself is multiplicatively independent.
    # Thus, the maximum possible size is 12.
    
    # We can simply return the primes in range(2, 41).
    # This is a valid solution of size 12, which is the theoretical maximum.
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [n for n in range(2, 41) if is_prime(n)]
    
    # Verify distinct subset products for safety, though theoretically guaranteed for primes.
    # For k=12, 2^12 = 4096 products. This is very fast.
    products = set()
    valid = True
    # Generate all subset products
    # Bitmask iteration for 12 elements
    k = len(primes)
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= primes[i]
        if prod in products:
            valid = False
            break
        products.add(prod)
    
    if not valid:
        # Fallback to a known smaller independent set if verification fails (should not happen)
        return [2, 3, 5, 7]
        
    return primes

# EVOLVE-BLOCK-END
