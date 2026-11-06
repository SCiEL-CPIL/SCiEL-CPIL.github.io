import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inverses = [0] * MAX_DENOM
    for i, d in enumerate(denominators):
        inverses[i] = pow(d, -1, MOD)
        
    max_mask = 1 << MAX_DENOM
    
    # Find the shortest subset for each residue
    best_mask = [None] * MOD
    best_len = [MAX_DENOM + 1] * MOD
    
    # Precompute sum and length for all masks using DP
    mask_sum = [0] * max_mask
    mask_len = [0] * max_mask
    for mask in range(1, max_mask):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        mask_sum[mask] = (mask_sum[prev] + inverses[idx]) % MOD
        mask_len[mask] = mask_len[prev] + 1
    
    # Iterate through all non-empty masks
    for mask in range(1, max_mask):
        s = mask_sum[mask]
        l = mask_len[mask]
        if l < best_len[s]:
            best_len[s] = l
            best_mask[s] = mask
    
    # Check if all residues are covered
    for r in range(MOD):
        if best_mask[r] is None:
            # Should not happen, but fallback
            best_mask[r] = 1  # denominator 1
            best_len[r] = 1
    
    # Construct the final result
    results = [None] * MOD
    for r in range(MOD):
        mask = best_mask[r]
        combo = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            combo.append(denominators[idx])
            m ^= lsb
        results[r] = combo
        
    return results
