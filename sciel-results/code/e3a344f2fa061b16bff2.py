"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _powerful_part_segmented():
    """
    Compute the powerful part for every integer in [1, N] using a segmented sieve.
    powerful_part(n) = product of p^e for each prime p where exponent e >= 2 in factorization of n.
    """
    N = 1_000_000
    # We need to find, for each n, the product of p^e where e >= 2.
    # Strategy: for each prime p, for each power p^k (k>=2), find all multiples of p^k in [1,N].
    # For each such multiple m, we need to determine the exact exponent of p in m.
    # The powerful part contribution from p to m is p^e where e is the exact exponent.
    # But we only include it if e >= 2.
    
    # More efficient: for each prime p, iterate through powers p^2, p^3, ...
    # For m divisible by p^k but not p^(k+1), the exponent is exactly k.
    # So for each prime p, for each k >= 2:
    #   Let pk = p^k, pk1 = p^(k+1)
    #   For m in [1, N] where m % pk == 0 and m % pk1 != 0:
    #     Add pk to the powerful part of m (multiply)
    #   For m in [1, N] where m % pk1 == 0:
    #     Add pk1 to the powerful part of m (multiply) ... wait, no.
    
    # Let me reconsider. For a number m, if p^e || m (exact exponent e), then:
    # - If e >= 2, the powerful part includes p^e.
    # - If e < 2, the powerful part includes nothing from p.
    
    # So for each prime p, I need to find the exact exponent for each multiple.
    # Alternative approach: 
    #   powerful = array of 1s of size N+1
    #   For each prime p:
    #     p2 = p*p
    #     For m = p2, 2*p2, 3*p2, ... <= N:
    #       Count how many times p divides m: e
    #       If e >= 2: powerful[m] *= p^e
    #     But this is O(N/p^2 * log_p(N)) per prime, which could be slow.
    
    # Better: For each prime p, process powers p^2, p^3, p^4, ...
    #   For k = 2, 3, 4, ... while p^k <= N:
    #     pk = p^k
    #     pk1 = p^(k+1)
    #     For m in [1, N] where m % pk == 0:
    #       These m have exponent at least k
    #     For m in [1, N] where m % pk1 == 0:
    #       These m have exponent at least k+1
    #     So m where m % pk == 0 and m % pk1 != 0 have exact exponent k.
    #     For those, multiply powerful[m] by pk.
    
    # This can be done with vectorized operations.
    
    powerful = np.ones(N + 1, dtype=np.int64)
    
    # Get all primes up to N
    # Simple sieve for primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        pk = p2
        k = 2
        while pk <= N:
            pk1 = pk * p
            if pk1 > N:
                # All multiples of pk in [1, N] have exact exponent k
                # because pk1 > N means no multiple of pk can be a multiple of pk1
                # Wait, that's not right. m = 2*pk could have higher exponent if 2 is divisible by p.
                # No, p is prime, so 2*pk has the same p-exponent as pk unless 2 == p.
                # Actually, m % pk == 0 means m = j*pk for some j. The exponent of p in m 
                # is k + (exponent of p in j). So we need to check j's divisibility by p.
                
                # Hmm, this approach is getting complicated. Let me use a different method.
                
                # For each multiple m of pk: m = j * pk
                # The exponent of p in m is k + v_p(j)
                # So I need to find v_p(j) for each j.
                
                # This is getting complex. Let me just do it directly for each multiple.
                pass
            break  # placeholder
    
    # Let me use a cleaner approach:
    # For each n in [1, N], factorize it and compute the powerful part.
    # But factoring each number individually is too slow.
    
    # Alternative: use a sieve-like approach.
    # powerful[n] starts at 1.
    # For each prime p:
    #   For each power pk = p^k (k >= 2) with pk <= N:
    #     For each multiple m of pk:
    #       Determine exact exponent e of p in m.
    #       If e >= 2: powerful[m] *= p^e
    #     But we should only do this once per (m, p) pair.
    
    # Cleaner: For each prime p, for each multiple m of p^2:
    #   Count the exact exponent e of p in m.
    #   powerful[m] *= p^e
    
    # To count exact exponent: while m % p == 0: e++, m //= p
    # But doing this for each (p, m) pair is O(N log N) total, which might be okay.
    
    # Let me try a vectorized approach per prime.
    powerful = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # Find all multiples of p2 in [1, N]
        # m = p2, 2*p2, 3*p2, ..., floor(N/p2)*p2
        # For each such m, compute exact exponent of p in m.
        
        # m = j * p2, where j = 1, 2, ..., floor(N/p2)
        # v_p(m) = 2 + v_p(j)
        # So we need v_p(j) for j = 1, 2, ..., floor(N/p2)
        
        max_j = N // p2
        
        # Compute v_p(j) for j = 1..max_j
        # v_p(j) = 0 for most j
        # For j divisible by p: v_p(j) >= 1
        # For j divisible by p^2: v_p(j) >= 2
        # etc.
        
        # Let's compute the exponent for each j.
        # exponents = np.zeros(max_j + 1, dtype=np.int32)
        # For k = 1, 2, ... while p^k <= max_j:
        #   For multiples of p^k: exponents[multiple] += 1
        
        # Then v_p(j) = exponents[j]
        # And v_p(m) = 2 + exponents[j] where m = j * p2
        
        # The powerful part contribution from p to m is p^(2 + exponents[j]) if 2 + exponents[j] >= 2
        # which is always true since exponents[j] >= 0.
        
        # So powerful[m] *= p^(2 + exponents[j])
        
        # Let me compute this.
        if max_j < 1:
            continue
            
        exponents = np.zeros(max_j + 1, dtype=np.int32)
        pk = p
        k = 1
        while pk <= max_j:
            # Add 1 to exponents for all multiples of pk
            exponents[pk::pk] += 1
            pk *= p
            k += 1
        
        # Now for j = 1..max_j, v_p(m_j) = 2 + exponents[j] where m_j = j * p2
        # powerful[m_j] *= p^(2 + exponents[j])
        
        # We need to compute p^(2 + exponents[j]) for each j.
        # This is p^2 * p^(exponents[j])
        
        # Let's compute p_powers[j] = p^(exponents[j])
        # Then multiply powerful[j*p2] by p2 * p_powers[j]
        
        # To compute p^(exponents[j]) for all j, we can use:
        # p_powers = np.ones(max_j + 1, dtype=np.int64)
        # For each k where exponents[j] >= k: p_powers[j] *= p
        
        # Or: p_powers[j] = p ** exponents[j], but this might overflow or be slow.
        
        # Let's do it iteratively.
        p_powers = np.ones(max_j + 1, dtype=np.int64)
        pk = p
        k = 1
        while pk <= max_j:
            # For all j where exponents[j] >= k, multiply p_powers[j] by p
            # These are multiples of pk
            p_powers[pk::pk] *= p
            pk *= p
            k += 1
        
        # Now p_powers[j] = p^(exponents[j])
        # Contribution to powerful[j*p2] is p2 * p_powers[j]
        
        # indices = np.arange(1, max_j + 1) * p2
        # contributions = p2 * p_powers[1:max_j+1]
        # powerful[indices] *= contributions
        
        # But we need to be careful about overflow. Let's use int64.
        
        indices = np.arange(1, max_j + 1, dtype=np.int64) * p2
        contributions = (p2 * p_powers[1:max_j + 1]).astype(np.int64)
        
        # Check for overflow? For now, assume it's okay.
        powerful[indices] *= contributions
    
    return powerful

def construct():
    start_time = time.time()
    deadline = 85.0  # seconds, safety margin before 90s limit
    
    N = 1_000_000
    MIN_N = 10_000
    
    # Compute powerful parts
    powerful = _powerful_part_segmented()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return MIN_N
    
    # Find n in [10000, 1000000] that maximizes powerful[n] * powerful[n+1]
    # We need n+1 <= N, so n <= N-1 = 999999
    # But the range is [10000, 1000000], and n+1 must be <= 1000000, so n <= 999999.
    
    products = powerful[MN_N:N] * powerful[MN_N+1:N+1]
    
    best_idx = np.argmax(products)
    best_n = MN_N + best_idx
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_n
    
    return int(best_n)
# EVOLVE-BLOCK-END
