"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    # We need a set S of distinct integers such that:
    # 210 < x <= 10210 for all x in S
    # |S| <= 100
    # product(S) * 210 is a perfect square
    # Minimize max(S) - 210
    
    # Strategy:
    # We model this as finding a subset of {211, ..., M} whose product has the same square-free part as 210.
    # This is equivalent to finding a subset whose XOR of prime-exponent parity vectors equals the vector of 210.
    # We iterate M from 211 upwards. For each M, we check if the target vector is in the span of the vectors for 211..M.
    # We use Gaussian elimination over GF(2) with bitmasks for efficiency.
    # We track the combination of original indices to reconstruct the subset.
    # We ensure the subset size is <= 100.
    
    # 1. Sieve of Eratosthenes for primes up to 10210.
    limit = 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Map prime to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Compute square-free kernel bitmask for each number in 211..10210
    # kernel[x] = bitmask
    kernel = [0] * (limit + 1)
    for x in range(2, limit + 1):
        mask = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                mask |= (1 << prime_to_idx[p])
        if temp > 1:
            # temp is a prime
            mask |= (1 << prime_to_idx[temp])
        kernel[x] = mask
    
    # Target mask for 210
    target_mask = kernel[210]
    
    # 3. Iterate M from 211 to limit
    # Maintain basis
    # basis[i] stores the vector with leading bit i.
    # And combo[i] stores the index mask for that vector.
    
    basis = [None] * num_primes
    combo = [0] * num_primes
    
    def add_to_basis(vec, idx_mask):
        # Returns True if vec was added to basis, False if it was reduced to 0 (dependent)
        v = vec
        c = idx_mask
        while v:
            # Find highest set bit
            hb = v.bit_length() - 1
            if basis[hb] is None:
                basis[hb] = v
                combo[hb] = c
                return True
            else:
                v ^= basis[hb]
                c ^= combo[hb]
        return False
    
    def is_in_span(target):
        # Returns (bool, index_mask)
        v = target
        c = 0
        while v:
            hb = v.bit_length() - 1
            if basis[hb] is None:
                return False, 0
            v ^= basis[hb]
            c ^= combo[hb]
        return True, c
    
    # Iterate M
    for M in range(211, limit + 1):
        # Add M to the system
        idx_mask = 1 << (M - 211)  # Bit position for index M
        
        add_to_basis(kernel[M], idx_mask)
        
        found, sol_mask = is_in_span(target_mask)
        if found:
            # Extract the solution
            selected = []
            m = sol_mask
            offset = 211
            i = 0
            while m:
                if m & 1:
                    selected.append(offset + i)
                m >>= 1
                i += 1
            
            # Check constraints
            if len(selected) <= 100:
                # Verify the condition
                prod = 1
                for x in selected:
                    prod *= x
                total = prod * 210
                # Check if perfect square
                root = math.isqrt(total)
                if root * root == total:
                    return selected
                else:
                    pass
            else:
                pass

    # Fallback: return a known valid solution
    # S = [840]
    return [840]

# EVOLVE-BLOCK-END
