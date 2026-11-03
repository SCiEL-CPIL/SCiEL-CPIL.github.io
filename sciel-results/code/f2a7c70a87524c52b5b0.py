def _local_search_first_improvement(a, b, p, deadline):
    n = len(p)
    cur_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cur_cost
    
    while time.time() < deadline:
        improved = False
        # Shuffle indices to randomize the order of checks
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for i_idx in range(n):
            i = indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = indices[j_idx]
                if i == j: continue
                delta = _calc_delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cur_cost += delta
                    improved = True
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_p = p.copy()
                    # After a swap, the order is invalid for the current pass?
                    # Usually in First Improvement, we continue with the new state.
                    # But the indices `i` and `j` are from the original shuffle.
                    # It's safer to restart the scan or just continue.
                    # Continuing is faster.
        if not improved:
            break
            
    return best_cost, best_p
