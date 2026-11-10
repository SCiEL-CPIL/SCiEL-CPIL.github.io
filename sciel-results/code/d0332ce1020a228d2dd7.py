import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 5.0

    # Precompute modular inverses for denominators 1..10
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)

    # Precompute sum and size for every mask from 0 to 1023 using DP
    mask_sum = [0] * 1024
    mask_size = [0] * 1024

    for mask in range(1, 1024):
        lsb = mask & -mask
        i = lsb.bit_length() - 1  # index 0..9 corresponds to denominator i+1
        prev_mask = mask ^ lsb
        denom = i + 1
        mask_sum[mask] = (mask_sum[prev_mask] + inv[denom]) % 101
        mask_size[mask] = mask_size[prev_mask] + 1

    # For each residue, collect all masks that map to it, sorted by size
    masks_by_residue = [[] for _ in range(101)]
    for mask in range(1, 1024):
        s = mask_sum[mask]
        sz = mask_size[mask]
        masks_by_residue[s].append((sz, mask))
    
    # Sort each residue's masks by size (then by mask for determinism)
    for r in range(101):
        masks_by_residue[r].sort(key=lambda x: (x[0], x[1]))

    # The current approach: for each residue, pick the smallest mask
    # This gives us a baseline solution. Let's compute the max row length.
    baseline_max = 0
    baseline_masks = [0] * 101
    for r in range(101):
        if masks_by_residue[r]:
            baseline_max = max(baseline_max, masks_by_residue[r][0][0])
            baseline_masks[r] = masks_by_residue[r][0][1]
        else:
            # Should not happen since every residue is covered
            baseline_max = float('inf')
            baseline_masks[r] = 0

    # The problem is small enough that we can try to improve the solution
    # by checking if we can reduce the maximum row length.
    # The maximum row length is the maximum over all residues of the size of the mask chosen.
    # We want to minimize this maximum.
    
    # Since the problem is very small, we can try a local search or a more careful assignment.
    # But first, let's see if the baseline is already optimal.
    
    # The minimum possible maximum row length is at least 1 (since all rows must be non-empty).
    # Can we achieve a maximum row length of 1? That would require each residue to have a mask of size 1.
    # There are only 10 denominators, so only 10 residues can have size-1 masks. The other 91 residues
    # must have masks of size >= 2. So the maximum row length is at least 2.
    
    # Can we achieve a maximum row length of 2? That would require every residue to have a mask of size <= 2.
    # Let's check if every residue has a mask of size <= 2.
    
    max_size_needed = 0
    for r in range(101):
        if not masks_by_residue[r]:
            max_size_needed = float('inf')
            break
        min_sz = masks_by_residue[r][0][0]
        max_size_needed = max(max_size_needed, min_sz)
    
    # If max_size_needed is 2, then we can achieve a maximum row length of 2.
    # If it's larger, we need a larger maximum row length.
    
    # Let's compute the actual maximum row length for the baseline solution.
    # The baseline solution picks the smallest mask for each residue, so the maximum row length
    # is the maximum over all residues of the size of the smallest mask for that residue.
    # This is exactly max_size_needed.
    
    # So the baseline solution already achieves the minimum possible maximum row length.
    # We can't do better than this.
    
    # Let's construct the result lists.
    result = [[] for _ in range(101)]
    for r in range(101):
        mask = baseline_masks[r]
        terms = []
        for i in range(10):
            if mask & (1 << i):
                terms.append(i + 1)
        result[r] = terms
    
    return result
