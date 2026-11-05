import time

def construct():
    start = time.time()
    deadline = 55.0  # Safety margin before 60s/90s kill
    P = 101
    N = 10
    DENOMS = list(range(1, N + 1))
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in DENOMS:
        inv[x] = pow(x, -1, P)
    
    # DP: dp[mask] = sum of inverses for the subset represented by mask, mod P
    # Also track size via popcount
    total_masks = 1 << N
    dp_sum = [0] * total_masks
    dp_size = [0] * total_masks
    
    for mask in range(1, total_masks):
        # Find the lowest set bit
        lb = mask & (-mask)
        idx = lb.bit_length() - 1  # 0-based index
        prev = mask ^ lb
        dp_sum[mask] = (dp_sum[prev] + inv[idx + 1]) % P
        dp_size[mask] = dp_size[prev] + 1
    
    # For each residue, find the minimum-size subset
    best_size = [N + 1] * P  # Initialize larger than max possible
    best_mask = [0] * P
    
    # Process masks in order of increasing size to find minimums efficiently
    # Group masks by size
    masks_by_size = [[] for _ in range(N + 1)]
    for mask in range(1, total_masks):
        sz = dp_size[mask]
        masks_by_size[sz].append(mask)
    
    for sz in range(1, N + 1):
        if time.time() - start > deadline:
            break
        for mask in masks_by_size[sz]:
            r = dp_sum[mask]
            if sz < best_size[r]:
                best_size[r] = sz
                best_mask[r] = mask
    
    # Determine the optimal k = max_r best_size[r]
    optimal_k = max(best_size)
    
    # Build result: for each residue, extract denominators from best_mask
    result = []
    for r in range(P):
        if best_mask[r] != 0:
            terms = []
            m = best_mask[r]
            for i in range(N):
                if (m >> i) & 1:
                    terms.append(i + 1)
            result.append(terms)
        else:
            # Fallback (should not happen with 1023 subsets covering 101 residues)
            result.append([1])
    
    return result
