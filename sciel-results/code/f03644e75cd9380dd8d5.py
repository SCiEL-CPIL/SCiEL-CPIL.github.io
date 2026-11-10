import numpy as np

def construct():
    N = 300
    
    # 1. Initialization: Square Grid
    # This is distinct from the hexagonal lattice used in the previous version.
    s = 0.1
    k = 5
    xs = np.arange(-k*s, k*s + s, s)
    ys = np.arange(-k*s, k*s + s, s)
    X, Y = np.meshgrid(xs, ys)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Keep points well inside
    radii = np.linalg.norm(points, axis=1)
    points = points[radii < 0.9]
    
    # If we have too many, pick random N
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points if grid is too sparse
        while len(points) < N:
            p = np.random.uniform(-0.8, 0.8, 2)
            if np.linalg.norm(p) < 0.9:
                points = np.vstack([points, p])
                
    # 2. Simulated Annealing Optimization
    # This approach uses stochastic local search to escape local minima,
    # distinct from the deterministic repulsion force method.
    
    r = 0.05
    max_r = 0.054
    r_step = 0.0002
    
    best_points = points.copy()
    
    # To speed up energy calculation, we define it here
    def get_energy(pts, r_val):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        viol = 2 * r_val - dist
        pos_viol = np.maximum(0, viol)
        pair_e = np.sum(pos_viol**2)
        
        radii = np.linalg.norm(pts, axis=1)
        b_viol = radii - (1 - r_val)
        pos_b_viol = np.maximum(0, b_viol)
        bound_e = np.sum(pos_b_viol**2)
        
        return pair_e + bound_e

    while r < max_r:
        T = 0.05
        T_min = 0.001
        cooling = 0.999
        
        E = get_energy(points, r)
        
        for i in range(2000):
            idx = np.random.randint(N)
            old_p = points[idx].copy()
            
            # Move
            move = np.random.normal(0, T, 2)
            new_p = old_p + move
            
            points[idx] = new_p
            E_new = get_energy(points, r)
            dE = E_new - E
            
            # Acceptance criterion
            if dE < 0 or (dE < 50 * T and np.random.rand() < np.exp(-dE / T)):
                E = E_new
            else:
                points[idx] = old_p
                
            T *= cooling
            if T < T_min:
                T = T_min
                
        # Check if this configuration is valid for r
        # If energy is very low, it's a good packing for this r
        if E < 1e-6:
            best_points = points.copy()
        else:
            # If not valid, we might have degraded the solution.
            # We should probably revert to best_points for the next iteration?
            # Or just continue, hoping SA fixes it?
            # If we continue with an invalid state, the next r step will be harder.
            # Let's revert to the last best known valid configuration if we fail?
            # Actually, if E is not 0, the current points have overlaps.
            # Using them as start for next r (larger) is bad.
            # We should restart from best_points?
            # But best_points is for a smaller r.
            # Let's just break if we can't satisfy the current r?
            # No, maybe we just need more iterations.
            # For now, let's keep the current points but not update best_points.
            # And for the next iteration, we start from current points?
            # That's risky.
            # Let's start the next r iteration from best_points (last valid) 
            # but with a slightly perturbed state?
            # No, let's just break if we can't satisfy r.
            # The best_points holds the last valid configuration.
            break
            
        r += r_step
        
    return best_points
