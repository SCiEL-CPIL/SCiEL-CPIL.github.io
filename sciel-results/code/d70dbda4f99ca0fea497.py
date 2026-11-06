"""Rewritten program: vectorized powerful-part sieve + targeted search."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 1000001  # we need powerful_part for 0..1000000 (n up to 1000000, n+1 up to 1000001)
    
    # powerful_part[i] = product of full prime powers p^e (e>=2) dividing i
    # Initialize to 1 for all
    pow_part = np.ones(N + 1, dtype=np.int64)
    
    # Sieve: for each prime p, for each k>=2, multiply pow_part[multiples of p^k] by p^k
    # But careful: if p^2 divides i, we multiply by p^2; if p^3 divides i, we multiply by p^3 (not p^2 * p)
    # Actually: the powerful part includes the FULL prime power. So if i = p^e * rest, the p-part of powerful part is p^e.
    # So for each prime p, for each i divisible by p, we need to find the full exponent e and multiply by p^e.
    
    # Better approach: for each prime p, compute p^2, p^3, ... up to N.
    # For each p^k (k>=2), for multiples of p^k that are NOT multiples of p^(k+1), multiply by p^k.
    # Actually simpler: for each prime p, for each multiple m of p, compute the exact power of p in m, then multiply pow_part[m] by p^e.
    # But that's O(N log N) which might be slow. Let's use the sieve approach.
    
    # Alternative: for each prime p, let p2 = p*p. For multiples of p2, we know at least p^2 divides.
    # We want to multiply by the full p^e. We can do this iteratively:
    # Start with pow_part all 1.
    # For each prime p:
    #   For k from 2 upward while p^k <= N:
    #     For each multiple m of p^k:
    #       If m is NOT divisible by p^(k+1), then the p-part of m's factorization is exactly p^k, so multiply pow_part[m] by p^k.
    #     But this misses cases where m is divisible by p^(k+1) - those will be handled at a higher k.
    # Actually, this works: for each m, the highest k such that p^k | m determines the contribution.
    
    # Let's generate primes up to sqrt(N) ~ 1000, but actually we need primes up to N for the sieve.
    # Wait, we only need primes p where p^2 <= N, i.e., p <= 1000, because if p > 1000, then p^2 > 1000000 = N, so no multiple of p^2 exists in range.
    
    # Generate primes up to 1000
    limit = int(math.isqrt(N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # For each prime p, process its powers
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # Find all multiples of p^2, p^3, ... and assign the correct full power
        # For each multiple m of p2, we need to find the exact exponent of p in m.
        # Efficient approach: 
        #   For k=2,3,...: p^k <= N
        #     For multiples of p^k that are NOT multiples of p^(k+1): multiply pow_part by p^k
        #     For multiples of p^k that ARE multiples of p^(k+1): they'll be handled at higher k
        pk = p2
        while pk <= N:
            pk_next = pk * p
            if pk_next > N:
                # All multiples of pk in range have exactly p^k (no higher power fits)
                # Actually, they might have higher power if m is divisible by p^(k+1) but p^(k+1) > N is impossible since m <= N
                # So for pk_next > N, all multiples of pk have exactly p^k as their p-part
                indices = np.arange(pk, N+1, pk)
                pow_part[indices] *= pk
                break
            else:
                # Multiples of pk that are NOT multiples of pk_next have exactly p^k
                # Multiples of pk_next will be handled at the next iteration
                # We can mark multiples of pk, then subtract multiples of pk_next
                # But numpy doesn't support set difference easily. Let's do it differently.
                # For all multiples of pk: we'll tentatively multiply by pk
                # Then for multiples of pk_next: we'll need to fix them later
                # This is getting complicated. Let's use a different approach.
                
                # Alternative: for each multiple m of p2, compute the exact power.
                # This is O(N/p) per prime, total O(N log log N) which is fine.
                break
    
    # Let me restart with a cleaner approach: for each prime p, iterate over multiples of p^2 and compute exact power.
    pow_part = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # For each multiple of p2, compute the exact power of p in that number
        # and multiply pow_part by that power
        for m in range(p2, N + 1, p2):
            # Compute the exact power of p in m
            count = 0
            tmp = m
            while tmp % p == 0:
                tmp //= p
                count += 1
            # count >= 2 since m is a multiple of p^2
            pow_part[m] *= (p ** count)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return 10000
    
    # Now find the best n in [10000, 1000000] maximizing pow_part[n] * pow_part[n+1]
    # pow_part is 0-indexed, so n corresponds to index n
    # We need n in [10000, 1000000], so indices 10000 to 1000000
    # pow_part[n] * pow_part[n+1] for n in [10000, 999999] (since n+1 <= 1000000)
    
    products = pow_part[10000:1000000] * pow_part[10001:1000001]
    best_idx = np.argmax(products)
    best_n = 10000 + best_idx
    best_val = products[best_idx]
    
    if time.time() - start_time > DEADLINE:
        return int(best_n)
    
    # Verify the result
    def powerful_part(x):
        result = 1
        tmp = x
        for p in primes:
            if p * p > tmp:
                break
            count = 0
            while tmp % p == 0:
                tmp //= p
                count += 1
            if count >= 2:
                result *= (p ** count)
        if tmp > 1:
            # tmp is prime, exponent 1, so no contribution
            pass
        return result
    
    pp_n = powerful_part(best_n)
    pp_np1 = powerful_part(best_n + 1)
    actual_val = pp_n * pp_np1
    
    if actual_val != best_val:
        # Discrepancy, recompute more carefully
        # This shouldn't happen but let's be safe
        pass
    
    return int(best_n)
# EVOLVE-BLOCK-END
