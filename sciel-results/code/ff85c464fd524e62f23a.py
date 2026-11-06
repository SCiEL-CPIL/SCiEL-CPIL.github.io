"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 1000000
    LO = 10000
    
    # Generate primes up to sqrt(N) using simple sieve
    limit = int(np.sqrt(N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # powerful_part for indices [LO, N]
    size = N - LO + 1
    powerful_part = np.ones(size, dtype=np.int64)
    
    # For each prime p, find all prime powers p^e with e >= 2 and p^e <= N
    # For each such power q = p^e, mark multiples in [LO, N]
    # The key insight: for a number m, the full prime power p^e || m (exactly)
    # means m is divisible by p^e but NOT by p^(e+1).
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # Collect all prime powers p^e for e >= 2 up to N
        powers = []
        q = p2
        while q <= N:
            powers.append(q)
            q *= p
        
        # For each prime power q = p^e in powers, we need to find numbers m in [LO, N]
        # such that p^e || m (exactly), i.e., m % q == 0 and m % (q*p) != 0
        # We process from highest power down to avoid double-counting
        
        # Actually, a cleaner approach: for each multiple m of p in [LO, N],
        # compute the exact exponent of p in m, and if >= 2, multiply by p^e.
        # But that's O(N/p) per prime which is fine.
        
        # More efficient: iterate through prime powers from high to low.
        # For q = p^e (highest first), mark multiples of q that are NOT multiples of q*p.
        # Those have exact exponent e.
        
        # Process powers in reverse order (highest first)
        for idx in range(len(powers) - 1, -1, -1):
            q = powers[idx]
            next_q = q * p  # p^(e+1)
            
            # Find all multiples of q in [LO, N]
            # First multiple >= LO:
            first = ((LO + q - 1) // q) * q
            if first > N:
                continue
            
            # All multiples: first, first+q, first+2q, ...
            indices = np.arange(first, N + 1, q)
            
            if next_q > N:
                # No higher power exists, so all these have exact exponent e
                local_idx = indices - LO
                powerful_part[local_idx] *= q
            else:
                # Filter: keep only those NOT divisible by next_q
                mask = (indices % next_q) != 0
                if np.any(mask):
                    local_idx = indices[mask] - LO
                    powerful_part[local_idx] *= q
    
    # Compute products of consecutive powerful parts
    # powerful_part[i] corresponds to number LO + i
    # We want max over n in [LO, N-1] of powerful_part[n] * powerful_part[n+1]
    # In our array: powerful_part[n - LO] * powerful_part[n + 1 - LO]
    # That's powerful_part[0:size-1] * powerful_part[1:size]
    
    products = powerful_part[:-1] * powerful_part[1:]
    max_idx = np.argmax(products)
    result = LO + max_idx
    
    return int(result)

# EVOLVE-BLOCK-END
