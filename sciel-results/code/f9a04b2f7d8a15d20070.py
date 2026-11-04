def repair(params):
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    # Wall repair
    dist_wall = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    r = np.minimum(r, dist_wall)
    
    # Overlap repair: Iterative
    for _ in range(100):
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        D = np.sqrt(diff_x**2 + diff_y**2)
        r_sum = r[:, None] + r[None, :]
        np.fill_diagonal(D, np.inf)
        overlap = r_sum - D
        max_overlap = np.max(overlap)
        if max_overlap <= 1e-9:
            break
        
        # Find the pair with max overlap
        i, j = np.unravel_index(np.argmax(overlap), overlap.shape)
        if i == j:
            break
            
        # Reduce radii of i and j
        # Reduce by half the overlap
        reduction = max_overlap / 2.0
        r[i] -= reduction
        r[j] -= reduction
        r[i] = max(0, r[i])
        r[j] = max(0, r[j])
        
    params[2::3] = r
    return params
