import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 75  # Safety margin for 90s limit
    
    def elapsed():
        return time.time() - t0
    
    MAXN = 1_000_000
    # We need pp[n] and pp[n+1] for n in [10000, 999999]
    # So we need pp up to 1000000
    
    # 1. Sieve of Eratosthenes to get primes
    # Using a standard Python sieve is fast enough for 1e6
    is_prime = bytearray(b'\x01') * (MAXN + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, MAXN + 1) if is_prime[i]]
    
    if elapsed() > DEADLINE:
        return 10000

    # 2. Compute powerful part P(m) for all m
    # P(m) = product of p^k where k = v_p(m) and k >= 2
    # We use an array of 64-bit integers.
    # Max powerful part is <= m <= 1e6, so 32-bit would suffice, but 64-bit is safer for intermediate products if we were doing sums, but here it's just assignment/multiplication.
    # Actually, P(m) divides m, so P(m) <= m.
    # We can store as int32 or int64. int64 is safer.
    
    pp = np.ones(MAXN + 1, dtype=np.int64)
    
    for p in primes:
        if elapsed() > DEADLINE:
            break
            
        p_sq = p * p
        if p_sq > MAXN:
            break
            
        # Iterate through powers of p: p^2, p^3, ...
        # For a number m, v_p(m) = k.
        # If k >= 2, P(m) includes p^k.
        # We want to set pp[m] *= p^k for the specific prime p.
        
        # Efficient way:
        # For each power p^k (k>=2), identify numbers where v_p(m) == k.
        # These are multiples of p^k but not p^(k+1).
        # Instead of checking "not multiple", we can update in a specific order or use a trick.
        
        # Trick:
        # Initialize contribution array for this prime to 1.
        # For each k >= 2 such that p^k <= MAXN:
        #   Let M = p^k.
        #   For all multiples m of M:
        #       If m is NOT a multiple of M*p, then v_p(m) == k.
        #       So pp[m] *= M.
        
        # This is O(N log N) roughly.
        # Let's try to vectorize the inner loop.
        
        pk = p_sq
        k = 2
        while pk <= MAXN:
            if elapsed() > DEADLINE:
                break
                
            next_pk = pk * p
            # Indices divisible by pk
            # We want to multiply pp[idx] by pk ONLY IF idx is NOT divisible by next_pk.
            
            # Range of indices: pk, 2*pk, 3*pk, ... <= MAXN
            # Vectorized approach:
            # Create array of indices: idx = np.arange(pk, MAXN+1, pk)
            # Mask: idx % next_pk != 0
            # pp[idx[mask]] *= pk
            
            # This creates temporary arrays. For small pk, this is fine.
            # For large pk, the array is small.
            
            idx = np.arange(pk, MAXN + 1, pk, dtype=np.int64)
            if len(idx) == 0:
                break
                
            # Check which are NOT divisible by next_pk
            # idx % next_pk != 0
            mask = (idx % next_pk) != 0
            valid_idx = idx[mask]
            
            if len(valid_idx) > 0:
                pp[valid_idx] *= pk
                
            pk = next_pk
            k += 1
            
    if elapsed() > DEADLINE:
        # Fallback: return a valid number, maybe scan a bit?
        # Just return 10000 to be safe, or best found?
        # We haven't computed the best yet.
        # Let's do a quick scan of the first 1000 numbers if we have time, else 10000.
        if elapsed() < DEADLINE - 5:
             # Quick scan
             best_val = 0
             best_n = 10000
             # Convert pp to list for faster access? Or keep numpy.
             # Numpy scalar access is slow. Let's convert to Python list.
             pp_list = pp.tolist()
             for n in range(10000, min(100000, MAXN)):
                 prod = pp_list[n] * pp_list[n+1]
                 if prod > best_val:
                     best_val = prod
                     best_n = n
             return best_n
        return 10000

    # 3. Find max pp[n] * pp[n+1]
    # Vectorized product
    # pp is 1-indexed effectively (0 is unused/1)
    # We want n in [10000, MAXN-1]
    # prod[n] = pp[n] * pp[n+1]
    
    # Slice pp
    # n goes from 10000 to MAXN-1
    # pp[n] is pp[10000:MAXN]
    # pp[n+1] is pp[10001:MAXN+1]
    
    left = pp[10000:MAXN]
    right = pp[10001:MAXN+1]
    
    products = left * right
    
    # Find index of max
    # max_idx is relative to the slice
    max_idx = np.argmax(products)
    
    best_n = 10000 + max_idx
    best_val = products[max_idx]
    
    return int(best_n)
