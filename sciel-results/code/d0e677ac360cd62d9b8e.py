"""Optimized program to find long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate primes up to 10^6
    primes = sieve_primes(10**6)
    N = 200
    
    def check_window(s):
        """Check if phi(s), ..., phi(s+N-1) are all distinct."""
        # Initialize arrays
        phis = [s + i for i in range(N)]
        rems = [s + i for i in range(N)]
        
        max_num = s + N - 1
        sqrt_max = int(math.isqrt(max_num))
        
        # Iterate through primes
        for p in primes:
            if p > sqrt_max:
                break
            
            # Find indices i in [0, N-1] such that (s+i) is divisible by p
            # s+i = k*p  =>  i = k*p - s
            # 0 <= i < N  =>  s <= k*p < s+N
            # k_min = ceil(s/p)
            # k_max = floor((s+N-1)/p)
            
            k_min = (s + p - 1) // p
            k_max = (s + N - 1) // p
            
            if k_min > k_max:
                continue
                
            for k in range(k_min, k_max + 1):
                i = k * p - s
                if rems[i] % p == 0:
                    # Apply (1 - 1/p)
                    phis[i] -= phis[i] // p
                    # Remove all factors of p from rems
                    while rems[i] % p == 0:
                        rems[i] //= p
                        
        # Handle remaining prime factors
        for i in range(N):
            if rems[i] > 1:
                phis[i] -= phis[i] // rems[i]
                
        # Check distinctness
        return len(set(phis)) == N

    # Search for a valid window
    # Start from a moderate value to balance collision probability and computation cost
    s = 10**5
    
    # Linear scan
    while time.time() < deadline - 10:
        if check_window(s):
            return {'start': s, 'length': N}
        
        s += 1
        
        # If s becomes too large, the check gets slower.
        # We can add a random jump strategy if linear scan is too slow.
        # But for now, linear scan from 10^5 is fast.
        
        if s > 10**12 - N:
            break

    # If no run of 200 found, try to find a smaller one?
    # Or just return a valid small run.
    # Given the task, finding 200 is the goal.
    # Fallback:
    return {'start': 1, 'length': 1}

# EVOLVE-BLOCK-END
