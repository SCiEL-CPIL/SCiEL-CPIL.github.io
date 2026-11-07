def construct():
    start = time.time()
    deadline = start + 140.0

    best_h = np.full(N, 0.5, dtype=np.float64)
    best_val = _correlation_value(best_h)

    rng = np.random.default_rng(42)
    n = N

    def local_search(h, max_iters=20000):
        h = h.copy()
        current_val = _correlation_value(h)
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick random pair
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
            
            # Try to adjust h[i] and h[j] to keep sum constant
            # Random direction and magnitude
            direction = rng.choice([-1, 1])
            # Adaptive step size
            step = rng.uniform(0.005, 0.05)
            
            delta = direction * step
            new_hi = h[i] + delta
            new_hj = h[j] - delta
            
            if 0.0 <= new_hi <= 1.0 and 0.0 <= new_hj <= 1.0:
                h[i] = new_hi
                h[j] = new_hj
                new_val = _correlation_value(h)
                
                if new_val < current_val - 1e-12:
                    current_val = new_val
                    no_improve_count = 0
                else:
                    # Revert
                    h[i] -= delta
                    h[j] += delta
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            # If stuck, try a larger random jump
            if no_improve_count > 100:
                i, j = rng.integers(0, n, 2)
                if i != j:
                    delta = rng.uniform(0.1, 0.3) * rng.choice([-1, 1])
                    new_hi = h[i] + delta
                    new_hj = h[j] - delta
                    if 0.0 <= new_hi <= 1.0 and 0.0 <= new_hj <= 1.0:
                        h[i] = new_hi
                        h[j] = new_hj
                        new_val = _correlation_value(h)
                        if new_val < current_val:
                            current_val = new_val
                        else:
                            h[i] -= delta
                            h[j] += delta
                no_improve_count = 0

        return h, current_val

    # Phase 1: Random restarts
    while time.time() < deadline - 20:
        h_init = rng.uniform(0, 1, n)
        h_init = _project_to_feasible(h_init)
        h_res, val_res = local_search(h_init, max_iters=5000)
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()

    # Phase 2: Polish best solution
    if time.time() < deadline - 5:
        h_res, val_res = local_search(best_h, max_iters=50000)
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()

    best_h = _project_to_feasible(best_h)
    return best_h
