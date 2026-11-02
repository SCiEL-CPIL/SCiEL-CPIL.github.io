<<<<<<< SEARCH
def _try_aspect_ratio(W, H, n_circles=21, time_limit=25.0):
    """Try packing for given aspect ratio with multiple strategies."""
    start = time.time()
    if W + H < 1e-6:
        return None, 0.0
    scale = 2.0 / (W + H)
    W *= scale
    H *= scale
    
    best_circles = None
    best_value = 0.0
    
    # Strategy 1: Hex lattice init + smooth optimization
    circles0 = _hex_lattice_init(W, H, n_circles)
    circles_opt = _optimize_smooth(circles0, W, H, max_iter=300, beta=80.0,
                                    time_limit=time_limit * 0.4)
    val = _compute_value(circles_opt, W, H)
    if val > best_value:
        best_value = val
        best_circles = circles_opt.copy()
    
    if time.time() - start > time_limit * 0.6:
        return best_circles, best_value
    
    # Strategy 2: LP polish
    if best_circles is not None:
        circles_lp = _polish_with_lp(best_circles, W, H, iterations=4,
                                      time_limit=time_limit * 0.35)
        val = _compute_value(circles_lp, W, H)
        if val > best_value:
            best_value = val
            best_circles = circles_lp.copy()
    
    # Strategy 3: Grid init + smooth optimization
    if time.time() - start < time_limit * 0.8:
        circles_grid = []
        for i in range(n_circles):
            r_idx = i // 5
            c_idx = i % 5
            x = (c_idx + 0.5) * W / 5
            y = (r_idx + 0.5) * H / 5
            r = min(W / 10, H / 10) * 0.9
            circles_grid.append([x, y, r])
        circles_grid = np.array(circles_grid)
        circles_opt2 = _optimize_smooth(circles_grid, W, H, max_iter=200, beta=60.0,
                                         time_limit=time_limit * 0.2)
        val = _compute_value(circles_opt2, W, H)
        if val > best_value:
            best_value = val
            best_circles = circles_opt2.copy()
    
    return best_circles, best_value
=======
def _try_aspect_ratio(W, H, n_circles=21, time_limit=25.0):
    """Try packing for given aspect ratio with multiple strategies."""
    start = time.time()
    if W + H < 1e-6:
        return None, 0.0
    scale = 2.0 / (W + H)
    W *= scale
    H *= scale
    
    best_circles = None
    best_value = 0.0
    
    # Strategy 1: Hex lattice init + smooth optimization
    circles0 = _hex_lattice_init(W, H, n_circles)
    # We must ensure the optimizer doesn't take too long
    # Reduce max_iter to be safe
    circles_opt = _optimize_smooth(circles0, W, H, max_iter=150, beta=80.0,
                                    time_limit=time_limit * 0.3)
    # Enforce validity before evaluating value to avoid invalid "high" scores
    if _check_valid(circles_opt, W, H, tol=1e-5):
        val = _compute_value(circles_opt, W, H)
        if val > best_value:
            best_value = val
            best_circles = circles_opt.copy()
    else:
        # If invalid, try to make it valid and check
        circles_opt = _make_valid(circles_opt, W, H, tol=1e-5)
        val = _compute_value(circles_opt, W, H)
        if val > best_value:
            best_value = val
            best_circles = circles_opt.copy()
    
    if time.time() - start > time_limit * 0.5:
        return best_circles, best_value
    
    # Strategy 2: LP polish
    if best_circles is not None:
        circles_lp = _polish_with_lp(best_circles, W, H, iterations=3,
                                      time_limit=time_limit * 0.3)
        if _check_valid(circles_lp, W, H, tol=1e-5):
            val = _compute_value(circles_lp, W, H)
            if val > best_value:
                best_value = val
                best_circles = circles_lp.copy()
        else:
            circles_lp = _make_valid(circles_lp, W, H, tol=1e-5)
            val = _compute_value(circles_lp, W, H)
            if val > best_value:
                best_value = val
                best_circles = circles_lp.copy()
    
    # Strategy 3: Grid init + smooth optimization (only if time permits)
    if time.time() - start < time_limit * 0.7:
        circles_grid = []
        for i in range(n_circles):
            r_idx = i // 5
            c_idx = i % 5
            x = (c_idx + 0.5) * W / 5
            y = (r_idx + 0.5) * H / 5
            r = min(W / 10, H / 10) * 0.9
            circles_grid.append([x, y, r])
        circles_grid = np.array(circles_grid)
        circles_opt2 = _optimize_smooth(circles_grid, W, H, max_iter=100, beta=60.0,
                                         time_limit=time_limit * 0.2)
        if _check_valid(circles_opt2, W, H, tol=1e-5):
            val = _compute_value(circles_opt2, W, H)
            if val > best_value:
                best_value = val
                best_circles = circles_opt2.copy()
        else:
            circles_opt2 = _make_valid(circles_opt2, W, H, tol=1e-5)
            val = _compute_value(circles_opt2, W, H)
            if val > best_value:
                best_value = val
                best_circles = circles_opt2.copy()
    
    return best_circles, best_value
>>>>>>> REPLACE

