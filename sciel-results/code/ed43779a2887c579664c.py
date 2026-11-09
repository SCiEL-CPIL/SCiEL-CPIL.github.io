import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to N, as K will be <= N
PRIMES = _primes(N)

def construct():
    # Estimate upper bound for K
    # K ln K <= N * exp(-gamma)
    # Approx K ~ 450 for N=5000
    # We can just iterate down from N, but that's slow.
    # Let's start from a reasonable guess.
    
    # To be safe, we can binary search or linear scan from high to low.
    # Max possible K is N.
    # But we know it's around 450.
    
    # Let's try to find the largest K such that sieve_size(K) >= K.
    
    best_k = 0
    best_set = []
    
    # We can optimize by noticing that if K works, K-1 might work with a superset?
    # No, the residues change.
    # But we can just check K from an upper bound down.
    
    # Upper bound:
    # D(K) = prod(1-1/p).
    # N * D(K) >= K.
    # Let's just scan from 1000 down to 1.
    # 1000 is safe upper bound (log 1000 ~ 6.9, 1000*6.9 = 6900 > 2800).
    
    for K in range(1000, 0, -1):
        # Run sieve for this K
        valid_indices = list(range(N + 1))
        
        # Only process primes <= K
        relevant_primes = [p for p in PRIMES if p <= K]
        
        for p in relevant_primes:
            if not valid_indices:
                break
            
            # Group by residue
            counts = [0] * p
            for v in valid_indices:
                counts[v % p] += 1
            
            # Find min count
            min_count = min(counts)
            
            # Remove all v with v % p == argmin
            # To be efficient, we can just filter
            # But we need to know which residue was min.
            min_res = counts.index(min_count)
            
            valid_indices = [v for v in valid_indices if v % p != min_res]
            
        if len(valid_indices) >= K:
            best_k = K
            best_set = valid_indices[:K]
            break
            
    return best_set
