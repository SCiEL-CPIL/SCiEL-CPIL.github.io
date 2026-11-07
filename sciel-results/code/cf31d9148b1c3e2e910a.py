# EVOLVE-BLOCK-START
import math

def construct():
    N = 210
    MAX_VAL = 10210
    
    # Sieve of Eratosthenes for primes up to 10210
    limit = MAX_VAL
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Compute square-free kernel bitmask for each number
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
            mask |= (1 << prime_to_idx[temp])
        kernel[x] = mask
        
    target_mask = kernel[210]
    
    basis = [None] * num_primes
    combo = [0] * num_primes
    
    def add_to_basis(vec, idx_mask):
        v = vec
        c = idx_mask
        while v:
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
        v = target
        c = 0
        while v:
            hb = v.bit_length() - 1
            if basis[hb] is None:
                return False, 0
            v ^= basis[hb]
            c ^= combo[hb]
        return True, c
        
    for M in range(211, limit + 1):
        idx_mask = 1 << (M - 211)
        add_to_basis(kernel[M], idx_mask)
        
        found, sol_mask = is_in_span(target_mask)
        if found:
            selected = []
            m = sol_mask
            offset = 211
            i = 0
            while m:
                if m & 1:
                    selected.append(offset + i)
                m >>= 1
                i += 1
                
            if len(selected) <= 100:
                # Verify
                prod = 1
                for x in selected:
                    prod *= x
                total = prod * 210
                root = math.isqrt(total)
                if root * root == total:
                    return selected
                else:
                    # Fallback if verification fails (should not happen)
                    pass
            else:
                # If size > 100, continue searching (though unlikely for small M)
                pass
                
    return [840]

# EVOLVE-BLOCK-END
