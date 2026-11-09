import math
import time

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin for 90s limit
    target = 210
    max_val = 10210

    # --- Precomputation ---
    # Sieve for smallest prime factor (SPF)
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Map primes to indices for bitmasking
    prime_index = {}
    primes = []
    for i in range(2, max_val + 1):
        if spf[i] == i:
            prime_index[i] = len(primes)
            primes.append(i)
            
    def get_mask(n):
        """Compute the squarefree signature mask for n."""
        mask = 0
        temp = n
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                mask |= (1 << prime_index[p])
        return mask

    # Precompute masks for all numbers up to max_val
    masks = [0] * (max_val + 1)
    for x in range(2, max_val + 1):
        masks[x] = get_mask(x)

    target_mask = masks[target]

    # --- Search ---
    # basis[bit] = (vector_mask, frozenset_of_indices)
    basis = {}
    
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            # Fallback: Return a known valid solution if time runs out
            # 210 * 840 = 420^2
            return [840]

        v = masks[M]
        
        # Insert M into basis
        temp_v = v
        temp_indices = frozenset([M])
        
        while temp_v != 0:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_mask, b_indices = basis[p]
                temp_v ^= b_mask
                temp_indices = frozenset(temp_indices ^ b_indices)
            else:
                basis[p] = (temp_v, temp_indices)
                break

        # Check if target is in span
        t_v = target_mask
        t_indices = frozenset()
        
        while t_v != 0:
            p = t_v.bit_length() - 1
            if p in basis:
                b_mask, b_indices = basis[p]
                t_v ^= b_mask
                t_indices = frozenset(t_indices ^ b_indices)
            else:
                t_v = -1 # Not in span
                break
        
        if t_v == 0:
            # Solution found
            sol = list(t_indices)
            if len(sol) <= 100:
                # Verify
                prod = target
                for x in sol:
                    prod *= x
                if math.isqrt(prod) ** 2 == prod:
                    # Ensure types are correct
                    return [int(x) for x in sol]
                else:
                    continue
            else:
                # If solution is too large, we might need to search for a smaller one
                # But for small M, this is unlikely.
                pass

    # Fallback
    return [840]
