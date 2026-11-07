import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    base = 210
    max_value = 10210
    max_size = 100
    
    # 1. Sieve primes up to max_value
    limit = max_value
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    primes = [i for i in range(limit+1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Precompute square-free part bitsets for all n in (210, max_value]
    # sqf_bits[n] is a bitmask where bit i is 1 if prime[i] divides n to an odd power
    sqf_bits = [0] * (max_value + 1)
    for n in range(211, max_value + 1):
        temp = n
        bits = 0
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                bits |= (1 << prime_to_idx[p])
        if temp > 1:
            # temp is a prime factor
            bits |= (1 << prime_to_idx[temp])
        sqf_bits[n] = bits
        
    # Target bitset: sqf(210) = 2 * 3 * 5 * 7
    target_bits = 0
    for p in [2, 3, 5, 7]:
        target_bits |= (1 << prime_to_idx[p])
        
    # 3. Gaussian Elimination over GF(2)
    # basis: list of (pivot_bit_index, row_bits, combination_bits)
    # We maintain rows in reduced row echelon form (or just row echelon)
    # To check if target is in span, we reduce target using the basis.
    
    basis = [] # List of dicts or tuples: (pivot_idx, bits, comb)
    
    for M in range(211, max_value + 1):
        if time.time() > deadline:
            # Fallback: return a known valid solution if we have one, 
            # otherwise we might need to return a heuristic one.
            # But we should find the answer very quickly.
            # If we haven't, just return a large set that works?
            # Actually, we can just return [840] as a safe fallback if search fails.
            return [840]
            
        v = sqf_bits[M]
        idx = M - 211 # 0-based index for original numbers
        comb = 1 << idx
        
        # Reduce v by existing basis
        # We want to find if v is independent
        curr_v = v
        curr_comb = comb
        
        for i, (pivot, row_bits, row_comb) in enumerate(basis):
            if (curr_v >> pivot) & 1:
                curr_v ^= row_bits
                curr_comb ^= row_comb
                
        if curr_v != 0:
            # Find pivot for curr_v
            # Pivot is the highest set bit? Or lowest?
            # Standard GE: pivot is the bit position.
            # Let's use highest set bit for consistent ordering?
            # Actually, any fixed order works. Let's use highest bit index.
            pivot = curr_v.bit_length() - 1
            
            # Check if there is already a basis vector with this pivot
            # In our loop above, we reduced by ALL basis vectors.
            # So if curr_v != 0, its highest bit must not be a pivot of any existing row?
            # Wait, if we iterate basis in order of pivot, and reduce,
            # then the resulting curr_v has 0s in all pivot positions of existing rows.
            # So the highest set bit of curr_v is a new pivot.
            
            basis.append((pivot, curr_v, curr_comb))
            # Optional: reorder basis to keep pivots sorted for efficiency?
            # Not strictly necessary for correctness, but good for reduction speed.
            # Since we only add, and reduce against all, it's fine.
            # However, reduction speed depends on checking all basis vectors.
            # We can sort basis by pivot to break early?
            # If we sort by pivot (highest first), we can break if pivot < current bit.
            # Let's keep it simple first.
            
        # Check if target is in the span of the current basis
        # Reduce target_bits using the basis
        t_curr = target_bits
        t_comb = 0
        
        for i, (pivot, row_bits, row_comb) in enumerate(basis):
            if (t_curr >> pivot) & 1:
                t_curr ^= row_bits
                t_comb ^= row_comb
                
        if t_curr == 0:
            # Target is in the span
            # t_comb contains the indices of the original numbers
            # Extract indices
            indices = []
            c = t_comb
            while c:
                # Get lowest set bit
                lsb = c & -c
                idx_pos = lsb.bit_length() - 1
                indices.append(idx_pos + 211) # Convert back to number
                c ^= lsb
                
            if 1 <= len(indices) <= max_size:
                # Verify
                prod = base
                for x in indices:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return list(indices)
                else:
                    # Should not happen, but if it does, continue
                    continue
            else:
                # Solution size too large, continue searching for larger M?
                # No, if the minimal M gives a large solution, a larger M might give a smaller solution?
                # The problem asks to minimize max(selected).
                # If M is the max, the solution uses numbers <= M.
                # It is possible that for M, the only solution has size 101,
                # but for M+1, there is a solution of size 2.
                # However, the span only grows.
                # If T is in span of V_M, it is in span of V_{M+1}.
                # The set of solutions is the affine space T + NullSpace.
                # As M increases, NullSpace dimension increases, so we have more freedom to find a sparse solution.
                # So yes, we should continue if size > 100.
                continue
                
    # Fallback
    return [840]
