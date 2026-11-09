<<<<<<< SEARCH
def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    start_time = time.time()
    timeout = 280 # Safety margin
    
    p = prizes
    d = distance_matrix
    ml = max_len
    
    # Candidate heuristic generators
    # Note: H_ij depends on i and j.
    # d_j0 = d[:, 0]
    d_j0 = d[:, 0]
    
    def gen_1(p, d, d_j0, ml):
        # p_j / (d_ij + d_j0)
        # Numerator: p_j broadcast to rows
        # Denominator: d_ij + d_j0 broadcast to columns? No, d_j0 depends on j.
        # d_ij is (n, n). d_j0 is (n,). We want d_ij + d_j0[j].
        # d_j0[None, :] is (1, n). 
        denom = d + d_j0[None, :]
        return (p[None, :] + 1e-9) / (denom + 1e-9)
        
    def gen_2(p, d, d_j0, ml):
        # p_j / d_ij
        return (p[None, :] + 1e-9) / (d + 1e-9)
        
    def gen_3(p, d, d_j0, ml):
        # p_j / (d_ij + d_j0)^2
        denom = d + d_j0[None, :]
        return (p[None, :] + 1e-9) / (denom + 1e-9)**2
        
    def gen_4(p, d, d_j0, ml):
        # exp(- (d_ij + d_j0)) * p_j
        denom = d + d_j0[None, :]
        return (p[None, :] + 1e-9) * np.exp(-(denom + 1e-9))
        
    def gen_5(p, d, d_j0, ml):
        # p_j / (d_ij * (d_ij + d_j0))
        denom = d * (d + d_j0[None, :])
        return (p[None, :] + 1e-9) / (denom + 1e-9)
        
    def gen_6(p, d, d_j0, ml):
        # (p_j / d_ij) / (d_j0 + 1)  -- Penalize nodes far from start
        base = (p[None, :] + 1e-9) / (d + 1e-9)
        penalty = 1.0 / (d_j0[None, :] + 1.0)
        return base * penalty
        
    def gen_7(p, d, d_j0, ml):
        # p_j / (d_ij + 0.5 * d_j0)
        denom = d + 0.5 * d_j0[None, :]
        return (p[None, :] + 1e-9) / (denom + 1e-9)

    candidates = [gen_1, gen_2, gen_3, gen_4, gen_5, gen_6, gen_7]
    
    best_score = -np.inf
    best_matrix = None
    
    # Use a few seeds to estimate robustness
    # 5 seeds * 7 candidates = 35 ACO runs.
    # Should be fast enough.
    test_seeds = [42, 123, 999, 7, 314]
    
    for gen in candidates:
        if time.time() - start_time > timeout:
            break
            
        try:
            h_mat = gen(prizes, distance_matrix, d_j0, max_len)
            # Sanitize
            h_mat = np.where(np.isfinite(h_mat) & (h_mat > 0), h_mat, 1e-9)
            
            scores = []
            for seed in test_seeds:
                if time.time() - start_time > timeout:
                    break
                score = _run_aco(prizes, distance_matrix, max_len, h_mat, seed)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_matrix = h_mat
        except Exception:
            continue
            
    if best_matrix is None:
        # Fallback
        best_matrix = gen_1(prizes, distance_matrix, d_j0, max_len)
        best_matrix = np.where(np.isfinite(best_matrix) & (best_matrix > 0), best_matrix, 1e-9)
        
    return best_matrix
=======
def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    start_time = time.time()
    timeout = 290  # Safety margin before 300s limit

    p = prizes
    d = distance_matrix
    ml = max_len
    d_j0 = d[:, 0]

    # Start with a reasonable initial heuristic: prize / (d_ij + d_j0)
    denom = d + d_j0[None, :]
    h = (p[None, :] + 1e-9) / (denom + 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    # Use 2 seeds for faster evaluation during optimization
    opt_seeds = [42, 123]
    
    def evaluate(h_mat, seeds):
        scores = []
        for seed in seeds:
            if time.time() - start_time > timeout:
                break
            try:
                s = _run_aco(prizes, distance_matrix, max_len, h_mat, seed)
                scores.append(s)
            except Exception:
                scores.append(0.0)
        return np.mean(scores) if scores else 0.0
    
    # Coordinate ascent: try perturbing each (i,j) entry
    # Focus on entries where i != 0 and j != 0 (node 0 is start, not chosen as target)
    # Also, the heuristic[i,j] only matters when moving from i to j.
    # For each position, try scaling by factors
    
    best_score = evaluate(h, opt_seeds)
    best_h = h.copy()
    
    # Number of coordinate ascent sweeps
    max_sweeps = 50
    # Multipliers to try for each coordinate
    multipliers = [0.5, 0.7, 1.4, 2.0]
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > timeout - 60:
            break
        
        improved = False
        # Iterate over all (i, j) pairs where i != 0 and j != 0
        # We can skip i=0 row since ant starts at 0 but the heuristic[0,j] 
        # is used for the first move. Actually it IS used.
        # Let's include all i, j where j != 0 (j=0 is never a valid target)
        
        for i in range(n):
            if time.time() - start_time > timeout - 60:
                break
            for j in range(1, n):  # j starts at 1, node 0 is never a target
                if time.time() - start_time > timeout - 60:
                    break
                
                original_val = best_h[i, j]
                if original_val <= 0 or not np.isfinite(original_val):
                    original_val = 1e-9
                    best_h[i, j] = 1e-9
                
                for mult in multipliers:
                    new_val = original_val * mult
                    if new_val <= 0 or not np.isfinite(new_val):
                        continue
                    if new_val > 1e6:  # Cap to avoid extreme values
                        new_val = 1e6
                    
                    # Temporarily set and evaluate
                    best_h[i, j] = new_val
                    score = evaluate(best_h, opt_seeds)
                    
                    if score > best_score + 1e-6:
                        best_score = score
                        improved = True
                        # Keep this value
                        break
                    else:
                        # Revert
                        best_h[i, j] = original_val
        
        if not improved:
            break
    
    # Final sanity check
    best_h = np.where(np.isfinite(best_h) & (best_h > 0), best_h, 1e-9)
    
    # If we have time, do a final validation with more seeds
    if time.time() - start_time < timeout - 30:
        final_score = evaluate(best_h, [42, 123, 999, 7, 314])
        if final_score > 0:
            return best_h
    
    return best_h
>>>>>>> REPLACE
