"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 1000000
    # powerful[i] will store the powerful part of i
    # powerful[i] = product of p^e for all primes p where e = v_p(i) >= 2
    powerful = np.ones(N + 1, dtype=np.int64)
    
    # Sieve to find primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime, compute its contribution to the powerful part
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # For each multiple of p, find the exact power of p dividing it
        # We need to handle multiples of p^2, p^3, etc.
        # Strategy: for each multiple m of p, compute v_p(m)
        # But more efficiently: iterate over powers of p
        
        # For each power p^k (k>=2), the numbers divisible by p^k but not p^(k+1)
        # have exactly p^k in their powerful part (from prime p)
        # But numbers divisible by p^k also get contributions from higher powers if divisible
        
        # Simpler: for each multiple m of p, compute the highest power of p dividing m
        # and multiply that into powerful[m] if >= p^2
        
        # Vectorized approach:
        # multiples of p: p, 2p, 3p, ...
        # For each such multiple, we need v_p(m)
        
        # Alternative: for each power p^k with k>=2, for each multiple of p^k,
        # divide out p until we can't, then we know the exact power.
        
        # Most efficient: for each multiple m of p, compute m // (p^v_p(m))
        # and track v_p(m)
        
        # Let's do it differently:
        # For prime p, consider all multiples of p in [1, N]
        # For each multiple, we want the largest p^e dividing it
        # We can do: for k from 2 while p^k <= N:
        #   for each multiple of p^k:
        #     if not divisible by p^(k+1), then add p^k to powerful
        #     but this is tricky because a number divisible by p^5 gets p^5, not p^2*p^3
        
        # Correct approach: for each multiple m of p, compute e = v_p(m)
        # if e >= 2, multiply powerful[m] by p^e
        
        # To compute v_p(m) for all multiples of p efficiently:
        # multiples = np.arange(p, N+1, p)
        # For each multiple, we can repeatedly divide by p
        # But this is slow for large p
        
        # Better: use the fact that v_p(m) = k if m is divisible by p^k but not p^(k+1)
        # So for each k >= 1, the numbers with v_p(m) = k are multiples of p^k but not p^(k+1)
        
        # Let's compute for each k >= 2:
        pk = p * p  # p^2
        k = 2
        while pk <= N:
            next_pk = pk * p
            # Numbers divisible by pk but not by next_pk have v_p = k
            if next_pk <= N:
                # Multiples of pk
                multiples_of_pk = np.arange(pk, N + 1, pk)
                # Multiples of next_pk
                multiples_of_next = np.arange(next_pk, N + 1, next_pk)
                # Numbers with v_p = k are multiples_of_pk minus multiples_of_next
                # But we need to multiply powerful[m] by pk for those
                # Use a boolean mask
                mask = np.ones(N + 1, dtype=bool)
                mask[multiples_of_pk] = True
                mask[multiples_of_next] = False
                # Only consider indices >= 1
                mask[0] = False
                powerful[mask] *= pk
            else:
                # Numbers divisible by pk (and hence all higher powers) have v_p >= k
                # But since next_pk > N, all multiples of pk have v_p >= k
                # and we can't have v_p > k for any number <= N? No, we can.
                # Actually if next_pk > N, then no number <= N is divisible by next_pk,
                # so all multiples of pk have v_p >= k, but some may have v_p > k
                # Wait, that's not right. If next_pk > N, then no number in [1,N] is divisible by next_pk,
                # so any number divisible by pk has v_p >= k, but could have v_p = k, k+1, etc.
                # But we can't have v_p > k if p^(k+1) > N, so v_p = k for all multiples of pk
                multiples_of_pk = np.arange(pk, N + 1, pk)
                mask = np.zeros(N + 1, dtype=bool)
                mask[multiples_of_pk] = True
                mask[0] = False
                powerful[mask] *= pk
                break
            pk = next_pk
    
    # Now find the pair (n, n+1) with n in [10000, 999999] that maximizes powerful[n]*powerful[n+1]
    # n ranges from 10000 to 999999, so n+1 ranges from 10001 to 1000000
    start = 10000
    end = 999999  # inclusive, so n+1 = 1000000 max
    
    products = powerful[start:end+1] * powerful[start+1:end+2]
    
    best_idx = np.argmax(products)
    best_n = start + best_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
