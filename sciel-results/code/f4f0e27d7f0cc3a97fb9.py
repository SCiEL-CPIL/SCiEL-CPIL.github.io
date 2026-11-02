<<<<<<< SEARCH
def _local_optimize(x, y, r, max_iters=50):
    """Local optimization to improve a feasible configuration by expanding radii."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    current_obj = _objective(x, y, r)
    
    for it in range(max_iters):
        improved = False
        # Try to increase each radius slightly
        for i in range(N):
            # Try a few small increments
            for dr in [0.001, 0.0005, 0.0002]:
                r_test = r.copy()
                r_test[i] += dr
                if _feasible(x, y, r_test):
                    r = r_test
                    current_obj = _objective(x, y, r)
                    improved = True
                    break
        if not improved:
            break
    
    return x, y, r, current_obj
=======
def _local_optimize(x, y, r, max_iters=100):
    """Local optimization to improve a feasible configuration by expanding radii and adjusting positions."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    current_obj = _objective(x, y, r)
    
    for it in range(max_iters):
        improved = False
        
        # Try to increase each radius slightly
        for i in range(N):
            for dr in [0.002, 0.001, 0.0005, 0.0002]:
                r_test = r.copy()
                r_test[i] += dr
                if _feasible(x, y, r_test):
                    r = r_test
                    current_obj = _objective(x, y, r)
                    improved = True
                    break
        
        # Try small position adjustments to create more space
        for i in range(N):
            for dx in [-0.005, -0.002, 0.002, 0.005]:
                for dy in [-0.005, -0.002, 0.002, 0.005]:
                    x_test = x.copy()
                    y_test = y.copy()
                    x_test[i] += dx
                    y_test[i] += dy
                    # Clamp to bounds
                    x_test[i] = np.clip(x_test[i], r[i], 1 - r[i])
                    y_test[i] = np.clip(y_test[i], r[i], 1 - r[i])
                    
                    if _feasible(x_test, y_test, r):
                        # Check if this allows increasing any radius
                        best_dr = 0
                        best_j = -1
                        for j in range(N):
                            for dr2 in [0.001, 0.0005]:
                                r_test2 = r.copy()
                                r_test2[j] += dr2
                                if _feasible(x_test, y_test, r_test2):
                                    if dr2 > best_dr:
                                        best_dr = dr2
                                        best_j = j
                                    break
                        if best_j >= 0:
                            x = x_test
                            y = y_test
                            r[best_j] += best_dr
                            current_obj = _objective(x, y, r)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return x, y, r, current_obj
>>>>>>> REPLACE

