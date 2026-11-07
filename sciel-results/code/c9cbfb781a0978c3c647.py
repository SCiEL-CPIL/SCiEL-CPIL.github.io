"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _isqrt(n):
    if n < 0:
        raise ValueError
    if n == 0:
        return 0
    return math.isqrt(n)

def _prime_sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _factorize_with_primes(n, primes):
    """Return list of primes that divide n an odd number of times."""
    result = []
    temp = n
    for p in primes:
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            count += 1
            temp //= p
        if count % 2 == 1:
            result.append(p)
    if temp > 1:
        result.append(temp)
    return result

def construct():
    n_base = 210
    max_size = 100
    max_value = 10210
    
    # Primes up to sqrt(10210) ~ 101
    primes = _prime_sieve(150)
    
    # Target parity: 210 = 2 * 3 * 5 * 7
    # Use prime value as bit index
    target_bits = (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7)
    
    # Basis: dict mapping pivot_prime -> (vector_bits, composition_mask)
    basis = {}
    
    # Iterate M from 211 to max_value
    # i corresponds to number 211 + i
    for i in range(max_value - 210):
        num = 211 + i
        pf = _factorize_with_primes(num, primes)
        v = 0
        for p in pf:
            v |= (1 << p)
            
        u = v
        mask = 1 << i
        
        # Reduce u using basis
        while u != 0:
            p = u.bit_length() - 1
            if p in basis:
                b_vec, b_mask = basis[p]
                u ^= b_vec
                mask ^= b_mask
            else:
                break
                
        if u == target_bits:
            # Found a solution
            subset_indices = []
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                subset_indices.append(idx)
                m ^= lsb
                
            if len(subset_indices) <= max_size:
                result = [211 + idx for idx in subset_indices]
                # Verify
                prod = 210
                for x in result:
                    prod *= x
                s = _isqrt(prod)
                if s * s == prod:
                    return result
                else:
                    # Verification failed, should not happen
                    pass
            # If size > max_size, we ignore this solution and continue
            # We do NOT add to basis because u is target, not a new independent vector
            # (Target is not part of the span we are building for future reductions, 
            # we are just looking for it)
            
        elif u == 0:
            # Dependent, do nothing
            pass
        else:
            # Add to basis
            p = u.bit_length() - 1
            basis[p] = (u, mask)
            
    # Fallback if no solution found
    # This should not happen, but we must return something valid.
    # We can construct a large valid set.
    # For example, pick numbers such that their product * 210 is a square.
    # A simple way: pick 210 * k^2 for various k? No, distinct and > 210.
    # Let's just return a list of 100 numbers that work.
    # We can find one by random search or just return a known large solution.
    # For now, return an empty list is invalid.
    # Let's return a single number that makes it a square?
    # 210 * x = square => x = 210 * k^2.
    # x > 210 => k >= 1.
    # If k=1, x=210 (not > 210).
    # If k=2, x=210*4=840.
    # 210 * 840 = 210 * 210 * 4 = (210*2)^2 = 420^2. Square!
    # So [840] is a valid solution.
    # Max is 840. Value 840-210=630.
    # This is a valid fallback.
    return [840]
# EVOLVE-BLOCK-END
