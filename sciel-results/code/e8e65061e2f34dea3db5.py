"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] maximizing powerful_part(n) * powerful_part(n+1).
    
    powerful_part(n) is the product of p^e for each prime power p^e || n where e >= 2.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 1000001  # we need up to n+1 where n can be 1000000
    
    # Compute powerful part for all integers up to N using vectorized sieve
    # powerful_part[i] starts at 1 for all i
    powerful = np.ones(N + 1, dtype=np.int64)
    
    # Find all primes up to N using a simple sieve
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    # Sieve of Eratosthenes
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime, multiply powerful_part at indices divisible by p^2, p^3, ...
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # Find all indices divisible by p^2
        # For each such index, we need to find the full power p^e dividing it
        # and multiply powerful_part[index] by p^e
        
        # Strategy: for each power p^k (k>=2) that is <= N, 
        # for indices divisible by p^k but not p^(k+1), multiply by p^k
        # But it's easier to: for each index divisible by p^2, count how many times p divides it
        
        # More efficient: iterate over powers
        pk = p2
        while pk <= N:
            # Indices divisible by pk
            indices = np.arange(pk - 1, N + 1, pk)  # 0-indexed: pk-1, 2*pk-1, ... 
            # Wait, arange(pk, N+1, pk) gives pk, 2*pk, 3*pk, ...
            indices = np.arange(pk, N + 1, pk)
            
            # For these indices, check if they are also divisible by p*(pk) = p^(k+1)
            # If not, then the full power of p in the factorization is exactly pk
            # Actually, we want to multiply by the FULL power p^e where e is the exponent
            # of p in the factorization of the index.
            
            # Better approach: for each index, we want to multiply by p^e where 
            # p^e || index (i.e., p^e divides index but p^(e+1) doesn't)
            
            # Let's do it differently: for each power pk = p^k with k >= 2,
            # multiply powerful_part[i] by p for each i divisible by pk.
            # This way, if p^e || i, then i is divisible by p^2, p^3, ..., p^e,
            # so we multiply by p (e-1) times, giving p^(e-1). But we want p^e.
            
            # Hmm, let me reconsider. If p^e || i, then i is divisible by p^2, p^3, ..., p^e.
            # That's (e-1) multiples. So multiplying by p each time gives p^(e-1).
            # But we want p^e. So we should start from p^1? No, we only want e >= 2.
            
            # Alternative: for each index i, compute the exponent of p in i, say e.
            # If e >= 2, multiply powerful_part[i] by p^e.
            
            # Let's just do it directly for each index divisible by p^2:
            # count the exponent of p in the index.
            
            # For efficiency, let's process by power:
            # For pk = p^k, for indices divisible by pk but not p^(k+1), 
            # the exponent is exactly k, so multiply by pk.
            
            next_pk = pk * p
            if next_pk <= N:
                # Indices divisible by next_pk should NOT be processed here (they have higher exponent)
                # So we want indices divisible by pk but not by next_pk
                # This is tricky to do with simple arange
                
                # Let's just multiply all indices divisible by pk by p, 
                # and do this for all k >= 2. Then the total multiplier is p^(e-1) 
                # for an index with exponent e. We need p^e, so we're off by a factor of p.
                
                # Simpler: just multiply by p for each k >= 2 where pk divides the index.
                # For an index with exponent e, it's divisible by p^2, ..., p^e, so 
                # we multiply by p (e-1) times. Result: p^(e-1). 
                # We want p^e, so we need one more factor of p.
                
                # Alternative: multiply by p for each k >= 1 where pk divides the index, 
                # but only for k >= 2. That's the same thing.
                
                # Let me just do it the straightforward way: for each index divisible by p^2,
                # compute the full exponent and multiply by p^e.
                
                pass
            else:
                # This is the highest power, so all indices divisible by pk have exponent exactly k
                powerful[indices] *= pk
            
            pk = next_pk
    
    # Wait, the above logic is flawed. Let me redo this more carefully.
    
    # Reset
    powerful = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # For each index i divisible by p^2, find the exponent e of p in i,
        # and multiply powerful[i] by p^e.
        
        # To do this efficiently, iterate over powers p^k for k >= 2.
        # For each k, for indices divisible by p^k but not p^(k+1), multiply by p^k.
        
        pk = p2
        while pk <= N:
            next_pk = pk * p
            if next_pk <= N:
                # Indices divisible by pk but not by next_pk
                # These have exponent exactly k
                # Get all indices divisible by pk
                idx_div_pk = np.arange(pk, N + 1, pk)
                # Get all indices divisible by next_pk
                idx_div_next = np.arange(next_pk, N + 1, next_pk)
                # The difference is indices divisible by pk but not next_pk
                # We can use a mask
                mask = np.ones(len(idx_div_pk), dtype=bool)
                # But this is inefficient. Let's just multiply all by pk, 
                # and then for indices divisible by next_pk, we'll multiply by next_pk/pk = p later.
                
                # Actually, simpler approach: just multiply by p for each power level.
                # For an index with exponent e, it gets multiplied by p for levels 2, 3, ..., e.
                # That's (e-1) times, giving p^(e-1). We want p^e.
                
                # So let's just multiply by p for each level k >= 2 where p^k divides the index.
                # And then at the end, for each index, if it was multiplied at all, 
                # multiply by p one more time. But we don't track which indices were multiplied.
                
                # Cleanest: for each index i, if p^e || i with e >= 2, multiply by p^e.
                # Do it by: for k from 2 while p^k <= N: for i in range(p^k, N+1, p^k): 
                #   if i is NOT divisible by p^(k+1), then multiply powerful[i] by p^k.
                
                # But checking "not divisible by p^(k+1)" for each i is expensive.
                
                # Alternative: just multiply by p for each k >= 2 where p^k | i.
                # Then the total factor for an index with exponent e is p^(e-1).
                # We want p^e. So after processing all k, for each index that has any factor,
                # multiply by p. But we need to know which indices had e >= 2.
                
                # Let's just do it the naive but correct way for each prime:
                # For each i from p^2 to N, step p^2:
                #   count how many times p divides i, call it e.
                #   if e >= 2: powerful[i] *= p**e
                
                # This is O(N/p^2 * log_p(N)) per prime, which should be fine.
                
                pass
            
            # Multiply all indices divisible by pk by p
            # This contributes one factor of p for each level k >= 2
            idx = np.arange(pk, N + 1, pk)
            powerful[idx] *= p
            
            pk = next_pk
    
    # Now, for each index i, if the exponent of p in i is e >= 2, 
    # we've multiplied by p (e-1) times. We need p^e, so we're missing one factor of p.
    # But we don't know which indices had e >= 2 for each p.
    
    # This approach is getting complicated. Let me use a different strategy.
    
    # Reset and use a cleaner approach.
    powerful = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # For each index i divisible by p^2, compute the exponent of p in i.
        # Do this by dividing by p repeatedly.
        # For efficiency, process all indices divisible by p^2 at once.
        
        idx = np.arange(p2, N + 1, p2)
        if len(idx) == 0:
            continue
        
        # For each index in idx, find the exponent of p
        # We can do this by: temp = idx, count = 0
        # while temp % p == 0: temp //= p, count += 1
        # But this is a Python loop. Let's vectorize.
        
        # Alternative: for k = 2, 3, 4, ... while p^k <= N:
        #   indices_div_pk = np.arange(p^k, N+1, p^k)
        #   For indices in indices_div_pk, the exponent is at least k.
        #   We want to add a factor of p for each level k >= 2.
        #   So multiply powerful[indices_div_pk] by p.
        
        # After processing all k, an index with exponent e >= 2 has been multiplied 
        # by p for k = 2, 3, ..., e, which is (e-1) times. So we have p^(e-1).
        # We want p^e, so we need to multiply by p one more time for each such index.
        
        # But we can just include k = 1 in the loop? No, we only want e >= 2.
        
        # Let's just multiply by p for k = 1, 2, ..., e. That gives p^e.
        # But for k = 1, we'd be multiplying indices divisible by p, which includes 
        # indices with exponent 1. We don't want to include those.
        
        # OK, here's the clean solution:
        # For k = 2, 3, 4, ... while p^k <= N:
        #   For each index i divisible by p^k, multiply powerful[i] by p.
        # This gives p^(e-1) for an index with exponent e >= 2.
        # Then, for each index i that is divisible by p^2, multiply powerful[i] by p.
        # This gives p^e total.
        
        # So:
        pk = p2
        while pk <= N:
            idx = np.arange(pk, N + 1, pk)
            powerful[idx] *= p
            pk *= p
        
        # Now multiply by p one more time for all indices divisible by p^2
        idx = np.arange(p2, N + 1, p2)
        powerful[idx] *= p
    
    # Check time
    if time.time() > deadline:
        # Fallback: return a known good value or best found so far
        # For now, just return 10000
        return 10000
    
    # Compute product of consecutive powerful parts
    # prod[i] = powerful[i] * powerful[i+1] for i in [10000, 1000000]
    # We need i+1 <= 1000001, so i <= 1000000
    prod = powerful[10000:1000001] * powerful[10001:1000002]
    
    # Find the index of the maximum
    best_offset = np.argmax(prod)
    best_n = 10000 + best_offset
    
    # Verify the result is in range
    if best_n < 10000 or best_n > 1000000:
        best_n = 10000
    
    return int(best_n)

# EVOLVE-BLOCK-END
