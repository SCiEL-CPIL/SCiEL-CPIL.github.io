"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _squarefree_kernel(n):
    result = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            if cnt % 2 == 1:
                result *= d
        d += 1
    if temp > 1:
        result *= temp
    return result

def _is_perfect_square(x):
    if x < 0:
        return False
    r = math.isqrt(x)
    return r * r == x

def _sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    return [i for i in range(2, limit+1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    max_val = P['maximum_window']
    max_size = P['maximum_size']
    
    # Build kernel table
    kernel_table = {}
    for i in range(n + 1, max_val + 1):
        kernel_table[i] = _squarefree_kernel(i)
    
    # Build prime-to-bit mapping
    primes = _sieve_primes(max_val)
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    def kernel_to_bitmask(k):
        mask = 0
        for p, bit in prime_to_bit.items():
            if k % p == 0:
                mask |= (1 << bit)
        return mask
    
    target_mask = kernel_to_bitmask(n)  # kernel of 210 = 210
    
    val_to_mask = {}
    for v, k in kernel_table.items():
        val_to_mask[v] = kernel_to_bitmask(k)
    
    # Build mask -> sorted list of values
    mask_to_vals = {}
    for v, m in val_to_mask.items():
        if m not in mask_to_vals:
            mask_to_vals[m] = []
        mask_to_vals[m].append(v)
    for m in mask_to_vals:
        mask_to_vals[m].sort()
    
    # Initialize best with single-value solution
    # v = n * s^2 for s >= 2
    best_m = None
    best_subset = None
    
    # Single values
    if target_mask in mask_to_vals:
        for v in mask_to_vals[target_mask]:
            if best_m is None or v < best_m:
                best_m = v
                best_subset = [v]
    
    if best_m is None:
        # Fallback: just return [4*n] which always works
        return [4 * n]
    
    # Check pairs
    if time.time() < deadline - 2:
        best_pair_m = None
        best_pair = None
        
        for m1, vals1 in mask_to_vals.items():
            m2 = m1 ^ target_mask
            if m2 not in mask_to_vals:
                continue
            vals2 = mask_to_vals[m2]
            
            if m1 == m2:
                if len(vals1) >= 2:
                    v1, v2 = vals1[0], vals1[1]
                    m = max(v1, v2)
                    if best_pair_m is None or m < best_pair_m:
                        best_pair_m = m
                        best_pair = [v1, v2]
            else:
                for v1 in vals1:
                    if best_pair_m is not None and v1 >= best_pair_m:
                        break
                    for v2 in vals2:
                        if v2 == v1:
                            continue
                        m = max(v1, v2)
                        if best_pair_m is None or m < best_pair_m:
                            best_pair_m = m
                            best_pair = [v1, v2]
                        break
        
        if best_pair_m is not None and best_pair_m < best_m:
            best_m = best_pair_m
            best_subset = best_pair
    
    # Check triples
    if time.time() < deadline - 1:
        best_triple_m = None
        best_triple = None
        
        # Only consider values < best_m
        candidate_vals = [v for v in sorted(val_to_mask.keys()) if v < best_m]
        cand_mask_to_vals = {}
        for v in candidate_vals:
            m = val_to_mask[v]
            if m not in cand_mask_to_vals:
                cand_mask_to_vals[m] = []
            cand_mask_to_vals[m].append(v)
        for m in cand_mask_to_vals:
            cand_mask_to_vals[m].sort()
        
        masks_sorted = sorted(cand_mask_to_vals.keys())
        n_masks = len(masks_sorted)
        
        for i in range(n_masks):
            if time.time() >= deadline - 0.5:
                break
            m1 = masks_sorted[i]
            vals1 = cand_mask_to_vals[m1]
            for j in range(i, n_masks):
                m2 = masks_sorted[j]
                vals2 = cand_mask_to_vals[m2]
                m3 = target_mask ^ m1 ^ m2
                if m3 not in cand_mask_to_vals:
                    continue
                vals3 = cand_mask_to_vals[m3]
                
                if m1 == m2 == m3:
                    if len(vals1) >= 3:
                        v1, v2, v3 = vals1[0], vals1[1], vals1[2]
                        m = max(v1, v2, v3)
                        if best_triple_m is None or m < best_triple_m:
                            best_triple_m = m
                            best_triple = [v1, v2, v3]
                elif m1 == m2:
                    if len(vals1) >= 2 and len(vals3) >= 1:
                        v1, v2 = vals1[0], vals1[1]
                        v3 = vals3[0]
                        if v3 != v1 and v3 != v2:
                            m = max(v1, v2, v3)
                            if best_triple_m is None or m < best_triple_m:
                                best_triple_m = m
                                best_triple = [v1, v2, v3]
                elif m1 == m3:
                    if len(vals1) >= 2 and len(vals2) >= 1:
                        v1, v3 = vals1[0], vals1[1]
                        v2 = vals2[0]
                        if v2 != v1 and v2 != v3:
                            m = max(v1, v2, v3)
                            if best_triple_m is None or m < best_triple_m:
                                best_triple_m = m
                                best_triple = [v1, v2, v3]
                elif m2 == m3:
                    if len(vals2) >= 2 and len(vals1) >= 1:
                        v2, v3 = vals2[0], vals2[1]
                        v1 = vals1[0]
                        if v1 != v2 and v1 != v3:
                            m = max(v1, v2, v3)
                            if best_triple_m is None or m < best_triple_m:
                                best_triple_m = m
                                best_triple = [v1, v2, v3]
                else:
                    v1, v2, v3 = vals1[0], vals2[0], vals3[0]
                    if v1 != v2 and v2 != v3 and v1 != v3:
                        m = max(v1, v2, v3)
                        if best_triple_m is None or m < best_triple_m:
                            best_triple_m = m
                            best_triple = [v1, v2, v3]
        
        if best_triple_m is not None and best_triple_m < best_m:
            best_m = best_triple_m
            best_subset = best_triple
    
    # Verify
    prod = n
    for v in best_subset:
        prod *= v
    if not _is_perfect_square(prod):
        # Fallback
        return [4 * n]
    
    if not (len(best_subset) >= 1 and len(best_subset) <= max_size):
        return [4 * n]
    
    if not all(isinstance(v, int) and v > n and v <= max_val for v in best_subset):
        return [4 * n]
    
    if len(set(best_subset)) != len(best_subset):
        return [4 * n]
    
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
