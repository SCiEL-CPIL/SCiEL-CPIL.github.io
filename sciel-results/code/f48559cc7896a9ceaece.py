def _local_search_subgradient(h, target_sum, time_budget):
    """Local subgradient search using soft-min gradient."""
    n = len(h)
    best_h = h.copy()
    best_obj = 1.0 - _compute_cross_correlation_min(best_h)
    step_size = 0.05
    no_improve = 0
    beta_local = 30.0 # Moderate beta for smooth gradient
    
    while _time_remaining() > time_budget:
        h = best_h
        corrs = _compute_all_correlations_fft(h)
        
        # Compute soft-min gradient
        min_val = np.min(corrs)
        shifted = corrs - min_val
        # Avoid overflow/underflow
        max_shifted = np.max(shifted)
        exp_neg_beta = np.exp(-beta_local * (shifted - max_shifted))
        sum_exp_neg = np.sum(exp_neg_beta)
        weights = exp_neg_beta / sum_exp_neg
        
        grad = np.zeros(n)
        for k in range(1, n):
            w = weights[k-1] * (2.0 / n)
            if w > 1e-15:
                grad[:n-k] += w * h[k:]
                grad[k:] += w * h[:n-k]
        
        # We want to maximize the minimum correlation, so we move in direction of grad
        h_new = h + step_size * grad
        h_new = _project_to_simplex(h_new, target_sum)
        
        new_obj = 1.0 - _compute_cross_correlation_min(h_new)
        if new_obj < best_obj - 1e-14:
            best_obj = new_obj
            best_h = h_new
            no_improve = 0
            step_size = min(step_size * 1.05, 0.1)
        else:
            no_improve += 1
            step_size *= 0.8
            if no_improve > 30:
                # Try a different beta or break
                beta_local *= 0.5
                if beta_local < 5.0:
                    break
                step_size = 0.05
                no_improve = 0
    
    return best_h, best_obj
