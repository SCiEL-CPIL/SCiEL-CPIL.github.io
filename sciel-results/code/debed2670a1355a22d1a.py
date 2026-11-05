"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

TARGET = 123457

def linear_sieve(n):
    """Generate all primes up to n using linear sieve."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break
    return primes

def generate_smooth_numbers(primes_list, limit):
    """Generate all numbers composed only of the given primes, up to limit.
    Returns a set of such numbers."""
    smooth = set()
    smooth.add(1)
    # For each prime in primes_list, multiply existing smooth numbers by powers of that prime
    for p in primes_list:
        # Collect current smooth numbers to avoid modifying while iterating
        current = list(smooth)
        for s in current:
            v = s
            while v * p <= limit:
                v *= p
                if v not in smooth:
                    smooth.add(v)
    return smooth

def construct():
    # Precompute all primes up to TARGET
    primes = linear_sieve(TARGET)
    
    # Binary search on the index in primes for the smallest K (prime)
    # such that there exist two K-smooth numbers summing to TARGET
    lo, hi = 0, len(primes) - 1
    best_pair = None
    best_K_idx = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        K = primes[mid]
        # Generate all K-smooth numbers (numbers whose prime factors are all in primes[0..mid])
        smooth_set = generate_smooth_numbers(primes[:mid+1], TARGET)
        
        # Check if any two numbers in smooth_set sum to TARGET
        found_pair = None
        for a in smooth_set:
            b = TARGET - a
            if b in smooth_set and a >= 1 and b >= 1:
                found_pair = (a, b)
                break
        
        if found_pair is not None:
            best_pair = found_pair
            best_K_idx = mid
            hi = mid - 1  # Try smaller K
        else:
            lo = mid + 1  # Need larger K
    
    if best_pair is not None:
        return [best_pair[0], best_pair[1]]
    
    # Fallback: should not reach here for this target
    return [1, TARGET - 1]

# EVOLVE-BLOCK-END
