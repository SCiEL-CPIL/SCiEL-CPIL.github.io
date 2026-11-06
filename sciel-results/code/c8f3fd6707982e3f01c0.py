import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 1000000
    # We need n in [10000, 1000000], so n+1 can be up to 1000001.
    # Let's compute powerful parts for all numbers up to 1000001.
    max_val = 1000001
    
    # Initialize powerful part array with 1s
    # powerful_part[i] will hold the product of p^e for all prime powers p^e dividing i with e>=2
    pp = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve for primes
    # We need to iterate over primes p where p^2 <= max_val
    # p <= sqrt(max_val) ~ 1000
    limit = int(np.sqrt(max_val))
    
    # Standard sieve of Eratosthenes to find primes up to limit
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime p, compute p^2, p^3, ... and update pp
    for p in primes:
        p2 = p * p
        if p2 > max_val:
            break
        
        # The powerful part contribution from prime p is the highest power p^e dividing n,
        # IF e >= 2. If e < 2, contribution is 1.
        # Actually, the definition is: product of FULL prime powers p^e with exponent e>=2.
        # So if n is divisible by p^2 but not p^3, the factor is p^2.
        # If n is divisible by p^3, the factor is p^3.
        # Essentially, for each n, pp[n] *= p^e where e is the largest integer such that p^e divides n, provided e >= 2.
        
        # We can iterate through multiples of p^2.
        # For a multiple m of p^2, let m = k * p^2.
        # We need to determine the exact power of p dividing m.
        # Let v_p(m) be the exponent of p in m.
        # We want to multiply pp[m] by p^(v_p(m)) if v_p(m) >= 2, else 1.
        
        # Efficient way:
        # Iterate through powers of p: p^2, p^3, ...
        # For each power P = p^k (k>=2):
        #   For each multiple m of P that is NOT a multiple of p^(k+1):
        #     pp[m] *= P
        
        # Alternatively, we can just divide out the factor.
        # Let's try the direct approach:
        # For each p, let current_power = p^2.
        # While current_power <= max_val:
        #   next_power = current_power * p
        #   For each multiple m of current_power:
        #     if m is not a multiple of next_power (i.e., m % next_power != 0):
        #        pp[m] *= current_power
        #   current_power = next_power
        
        # This is correct but might be slow in Python loops.
        # Let's try to vectorize the inner loop.
        
        curr_pow = p * p
        while curr_pow <= max_val:
            next_pow = curr_pow * p
            
            # Indices divisible by curr_pow
            # We want to multiply pp[idx] by curr_pow for all idx divisible by curr_pow
            # BUT ONLY IF idx is NOT divisible by next_pow.
            # Because if idx is divisible by next_pow, it will be handled in the next iteration
            # with a higher power (next_pow), and we don't want to double count or undercount.
            # Wait, if idx is divisible by p^3, the powerful part includes p^3, not p^2.
            # So we should only apply p^2 to numbers whose p-adic valuation is exactly 2.
            
            # Multiples of curr_pow: curr_pow, 2*curr_pow, 3*curr_pow, ...
            # We need to exclude multiples of next_pow.
            
            # Vectorized approach:
            # Create array of indices: np.arange(curr_pow, max_val+1, curr_pow)
            # Filter out those divisible by next_pow.
            
            if next_pow > max_val:
                # All multiples of curr_pow have valuation exactly k (since next power exceeds range)
                # So we multiply all multiples of curr_pow by curr_pow
                idx = np.arange(curr_pow, max_val + 1, curr_pow)
                pp[idx] *= curr_pow
            else:
                idx = np.arange(curr_pow, max_val + 1, curr_pow)
                # Filter: idx % next_pow != 0
                mask = (idx % next_pow) != 0
                idx_filtered = idx[mask]
                pp[idx_filtered] *= curr_pow
            
            curr_pow = next_pow
            
            # Check time
            if time.time() > deadline:
                break
    
    if time.time() > deadline:
        # Return a safe default or best effort
        # Since pp might be partially computed, we should be careful.
        # But the sieve is deterministic and fast for small primes.
        # If we timed out, it's likely we didn't finish the sieve.
        # Let's just return 10000 as a fallback, or try to evaluate with what we have?
        # Evaluating with partial pp is dangerous.
        return 10000
    
    # Now find n in [10000, 1000000] that maximizes pp[n] * pp[n+1]
    # n is 1-indexed in our array? No, array indices are 0..max_val.
    # n ranges from 10000 to 1000000.
    # n+1 ranges from 10001 to 1000001.
    
    # We need to compute product for each n.
    # products = pp[10000:1000001] * pp[10001:1000002]
    
    # Note: max_val is 1000001.
    # pp has size max_val+1 = 1000002.
    # Indices 10000..1000000 for n.
    # Indices 10001..1000001 for n+1.
    
    # Let's slice carefully.
    # n_start = 10000
    # n_end = 1000000
    # vals_n = pp[10000:1000001]  # length 990001
    # vals_np1 = pp[10001:1000002] # length 990001
    
    # Check if we have enough data
    if len(pp) < 1000002:
        return 10000
        
    vals_n = pp[10000:1000001]
    vals_np1 = pp[10001:1000002]
    
    products = vals_n * vals_np1
    
    # Find index of max product
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    return int(best_n)

# EVOLVE-BLOCK-START
# The function construct is defined above.
# We need to ensure the entry point is construct.
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program:
# def construct():
#     return 10000

# My code defines construct.
# I should probably wrap it or just define it inside the block?
# The prompt says "Output the COMPLETE program in a single
