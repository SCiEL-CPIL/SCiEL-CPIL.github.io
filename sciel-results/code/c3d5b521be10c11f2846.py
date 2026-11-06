"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 1000000
    # powerful_part[i] will store the product of full prime powers p^e (e>=2) dividing i
    # Initialize to 1 for all
    powerful_part = np.ones(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to get primes up to sqrt(N) is not enough;
    # we need primes up to N, but for p^2 <= N, p <= sqrt(N) ~ 1000
    # Wait: if p^2 > N, then p cannot appear with exponent >= 2 in any number <= N.
    # So we only need primes p where p^2 <= N, i.e., p <= 1000.
    
    # Generate primes up to sqrt(N)
    limit = int(np.sqrt(N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each prime p, compute p^2, p^3, ... <= N
    # For each such power q = p^e, mark all multiples of q
    # powerful_part[m] *= p^e for the full power? No.
    # The powerful part is the product of FULL prime powers p^e where e is the actual exponent in m.
    # So for a given m and prime p, if p^e || m (exactly), then we multiply powerful_part[m] by p^e.
    # But we can't easily do "exactly" with simple marking.
    
    # Alternative approach:
    # For each prime p, let q = p^2, p^3, ..., p^k <= N.
    # For each multiple m of q, we know p^2 divides m.
    # But we need the FULL power. 
    # 
    # Better: for each prime p, iterate through multiples of p.
    # For each multiple m of p, count how many times p divides m (the exponent e).
    # If e >= 2, then powerful_part[m] *= p^e.
    #
    # But counting exponents for every multiple of every prime is O(N log log N * log N) which might be slow.
    #
    # More efficient:
    # For each prime p:
    #   q = p
    #   while q <= N:
    #     for m in range(q, N+1, q):
    #       if m % (q*p) != 0:  # p^k || m where k is current power
    #         if q >= p*p:  # exponent >= 2
    #           powerful_part[m] *= q
    #     q *= p
    #
    # This is still potentially slow in pure Python. Let's use NumPy vectorized operations.
    
    powerful_part = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # q will be p^e for e >= 2
        q = p2
        while q <= N:
            # Multiples of q: q, 2q, 3q, ...
            # For each such multiple m, check if m is NOT divisible by q*p
            # If not, then the exact power of p in m is e (where q = p^e)
            # So we multiply powerful_part[m] by q
            
            # Vectorized: indices = np.arange(q, N+1, q)
            # mask = (indices % (q*p)) != 0
            # powerful_part[indices[mask]] *= q
            
            # But this creates large arrays. Let's try it.
            step = q
            indices = np.arange(q, N + 1, step)
            if len(indices) == 0:
                break
            next_q = q * p
            if next_q > N:
                # All multiples of q have exactly exponent e (since no higher power exists)
                powerful_part[indices] *= q
            else:
                mask = (indices % next_q) != 0
                if np.any(mask):
                    powerful_part[indices[mask]] *= q
            q = next_q
    
    # Now compute powerful_part[i] * powerful_part[i+1] for i in [10000, 999999]
    # Find the index that maximizes this product
    products = powerful_part[10000:N] * powerful_part[10001:N+1]
    max_idx = np.argmax(products) + 10000
    
    return int(max_idx)

# EVOLVE-BLOCK-END
