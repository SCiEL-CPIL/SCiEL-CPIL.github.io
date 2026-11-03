<<<<<<< SEARCH
def _solve_aspect(w, h, t0):
    x = _make_grid(N, w, h)
    x = _relax_circles(x, w, h, iterations=500)

    if not _validate(x, w, h):
        return x

    # Optimize with SLSQP
    if time.time() < t0 + DEADLINE - 15.0:
        optimized = _slsqp_optimize(x, w, h, t0 + DEADLINE)
        if optimized is not None:
            return optimized

    return x
=======
def _penalty_objective(x, w, h, penalty_weight=100.0):
    """Penalty-based objective: maximize sum of radii with penalty for violations."""
    n = len(x) // 3
    pos = x[:2*n].reshape(n, 2)
    rad = x[2*n:]
    
    # Base objective: maximize sum of radii
    obj = -np.sum(rad)
    
    # Boundary violations
    for i in range(n):
        cx, cy, r = pos[i, 0], pos[i, 1], rad[i]
        # Left boundary
        if cx - r < 0:
            obj += penalty_weight * (cx - r) ** 2
        # Right boundary
        if cx + r > w:
            obj += penalty_weight * (cx + r - w) ** 2
        # Bottom boundary
        if cy - r < 0:
            obj += penalty_weight * (cy - r) ** 2
        # Top boundary
        if cy + r > h:
            obj += penalty_weight * (cy + r - h) ** 2
    
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rad[i] + rad[j]
            if dist < min_dist:
                obj += penalty_weight * (min_dist - dist) ** 2
    
    return obj


def _solve_aspect(w, h, t0):
    # Use hexagonal lattice for better initial packing
    x = _make_hex_grid(N, w, h)
    x = _relax_circles(x, w, h, iterations=1000)

    if not _validate(x, w, h):
        # Try square grid as fallback
        x = _make_grid(N, w, h)
        x = _relax_circles(x, w, h, iterations=1000)
        if not _validate(x, w, h):
            return x

    # Optimize using penalty-based L-BFGS-B (faster than SLSQP)
    if time.time() < t0 + DEADLINE - 20.0:
        x_flat = x.flatten()
        bounds = []
        for i in range(N):
            bounds.append((0, w))       # x position
            bounds.append((0, h))       # y position
            bounds.append((1e-6, min(w, h) / 2))  # radius
        
        try:
            res = minimize(
                _penalty_objective,
                x_flat,
                args=(w, h, 100.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < _penalty_objective(x_flat, w, h, 100.0):
                new_x = res.x.reshape(N, 3)
                new_x = _relax_circles(new_x, w, h, iterations=500)
                if _validate(new_x, w, h):
                    return new_x
        except Exception:
            pass

    return x
>>>>>>> REPLACE

