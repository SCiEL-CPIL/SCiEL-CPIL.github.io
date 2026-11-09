    b_indices = np.arange(1, MAX_N + 1)
    a1 = (3 * b_indices) // 2
    a2 = a1 + (3 * b_indices) % 2
    
    # Mask for valid a1
    mask1 = (a1 >= 1) & (a1 <= MAX_N) & (a1 != b_indices)
    # Mask for valid a2
    mask2 = (a2 >= 1) & (a2 <= MAX_N) & (a2 != b_indices) & (a2 != a1)
    
    # Get sigma values
    sig_b = sigma[b_indices]
    
    # Check equality for a1
    valid1 = mask1 & (sigma[a1] == sig_b)
    # Check equality for a2
    valid2 = mask2 & (sigma[a2] == sig_b)
    
    # Calculate errors for valid candidates
    # err = |2a - 3b| / (2b)
    # We only care about min err.
    
    # For a1:
    if np.any(valid1):
        idxs1 = np.where(valid1)[0]
        # This might be a lot of indices.
        # We can compute errors for all valid ones and take min.
        b_valid = b_indices[idxs1]
        a_valid = a1[idxs1]
        diffs = np.abs(2 * a_valid - 3 * b_valid)
        # Exclude diff == 0
        non_zero = diffs != 0
        if np.any(non_zero):
            diffs = diffs[non_zero]
            b_valid = b_valid[non_zero]
            errs = diffs / (2.0 * b_valid)
            min_idx = np.argmin(errs)
            best_err = errs[min_idx]
            best_pair = (int(a_valid[min_idx]), int(b_valid[min_idx]))
            
    # Similarly for a2
    