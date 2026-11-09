"""Search for n in [1000, 1000000] minimizing distinct small-prime factors in window."""
import time
import sys

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    max_n = 1000000
    window = 24
    
    # Precompute small-prime factorization masks for all numbers up to max_n + window
    # mask[i] is a bitmask where bit j is set if prime[j] divides i
    # We need to factor out ALL powers of small primes, so we just need to check divisibility
    # Actually, the task says "extract its complete factor composed of primes STRICTLY BELOW 24"
    # So for each n+i, we compute the part that is divisible only by primes < 24
    # Then count the number of DISTINCT primes in that factor across all 24 numbers
    
    # Wait, re-reading: "extract its complete factor composed of primes STRICTLY BELOW 24"
    # This means: for each m = n+i, compute the largest divisor of m that is a product of primes < 24
    # Then across all 24 values, count the number of distinct primes that appear in any of these factors
    
    # Actually, I think the objective is: for each n+i, compute f(n+i) = the part of n+i composed of primes < 24
    # Then the "number of distinct extracted factors" = number of distinct primes appearing in {f(n+1), ..., f(n+24)}
    
    # So we need to minimize the number of distinct small primes that divide at least one of n+1, ..., n+24
    
    # Precompute: for each number up to max_n + 24, which small primes divide it
    # Use a sieve-like approach
    max_val = max_n + window
    # bit_mask[m] = bitmask of small primes dividing m
    bit_mask = [0] * (max_val + 1)
    
    for j, p in enumerate(primes):
        bit = 1 << j
        for m in range(p, max_val + 1, p):
            bit_mask[m] |= bit
    
    # Now search for n in [1000, max_n] minimizing popcount of OR of bit_mask[n+1] through bit_mask[n+24]
    best_n = 1000
    best_count = 9  # maximum possible is 9
    
    # Linear scan
    start = 1000
    end = max_n
    step = 1
    
    # First pass: full scan
    for n in range(start, end + 1):
        if time.time() > deadline:
            break
        # Compute OR of bitmasks for n+1 to n+24
        mask_or = 0
        for i in range(1, window + 1):
            mask_or |= bit_mask[n + i]
        # Count bits
        count = bin(mask_or).count('1')
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                # Can't do better than 1
                return best_n
    
    return best_n
