def _max_abs_ap_sum_vec(signs):
    max_disc = 1
    for d in range(1, _N):
        idx = _IDX_BY_D[d]
        # idx shape (d, max_len)
        # Check bounds
        valid = idx < _N
        # Create a safe index array: replace invalid with 0
        safe_idx = np.where(valid, idx, 0)
        mat = signs[safe_idx]
        # Zero out invalid entries
        mat = np.where(valid, mat, 0)
        
        csum = np.cumsum(mat, axis=1)
        P = np.concatenate([np.zeros((mat.shape[0], 1), dtype=mat.dtype), csum], axis=1)
        
        MinPrefix = np.minimum.accumulate(P, axis=1)
        MaxSub = np.max(P - MinPrefix, axis=1)
        
        MaxPrefix = np.maximum.accumulate(P, axis=1)
        MinSub = np.min(P - MaxPrefix, axis=1)
        
        curr_max = np.max(np.maximum(MaxSub, -MinSub))
        if curr_max > max_disc:
            max_disc = curr_max
            
    return max_disc
