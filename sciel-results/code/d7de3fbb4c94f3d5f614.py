<<<<<<< SEARCH
def _local_search(h, c, max_iter=20000, deadline=130.0):
    """
    Fast local search: flip signs to reduce max|c|/(sum h)^2.
    Use delta updates to avoid full convolve. Track c and sum incrementally.
    """
    start = time.time()
    current_val, _ = _eval(h)
    s = float(np.sum(h))
    denom = s * s if s * s > 1e-12 else 1e-12
    current_max = float(np.max(np.abs(c)))
    
    # Track the lag positions of the current max
    # We'll recompute max only when needed, but for speed we track it.
    max_lag_pos = int(np.argmax(np.abs(c)))
    
    # Precompute for each idx the "neighbor sum" pattern to speed up delta
    # delta_c[L] = d*(h[idx+L] + h[idx-L]) for L != 0, 0 for L=0
    # = d * (2*h[idx] at L=0 corrected to 0, else h[idx+L]+h[idx-L])
    
    # We'll do random + greedy: try flipping, if it reduces value, accept.
    # To be fast, we compute the new max via updating c and taking max over the changed positions.
    
    np.random.seed(42)
    order = np.random.permutation(N)
    oi = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if oi >= N:
            oi = 0
            order = np.random.permutation(N)
        
        idx = int(order[oi])
        oi += 1
        target = -h[idx]
        d = target - h[idx]  # ±2
        
        # Compute delta_c
        Ls = np.arange(-(N - 1), N)
        valid1 = (idx + Ls >= 0) & (idx + Ls < N)
        valid2 = (idx - Ls >= 0) & (idx - Ls < N)
        term1 = np.where(valid1, h[idx + np.where(valid1, Ls, 0)], 0.0)
        term2 = np.where(valid2, h[idx - np.where(valid2, Ls, 0)], 0.0)
        delta = d * (term1 + term2)
        zero_pos = N - 1
        delta[zero_pos] = 0.0
        
        new_c = c + delta
        # Update sum: s_new = s + d
        new_s = s + d
        new_denom = new_s * new_s if new_s * new_s > 1e-12 else 1e-12
        new_max = float(np.max(np.abs(new_c)))
        new_val = new_max / new_denom
        
        if new_val < current_val - 1e-15:
            h[idx] = target
            c = new_c
            s = new_s
            denom = new_denom
            current_max = new_max
            current_val = new_val
    
    return h, current_val
=======

>>>>>>> REPLACE

