"""Circle packing in unit square n=32 using gradient-based optimization."""
import numpy as np
import time

N = 32
TIME_LIMIT = 110.0  # seconds, safety margin before 120s kill


def construct():
    start_time = time.time()
    
    # Precompute pair indices for efficiency
    pair_i, pair_j = np.triu_indices(N, k=1)
    n_pairs = len(pair_i)
    
    def pack_loss(params):
        """
        params: (N, 3) array of (x, y, r)
        Returns: (loss, valid) where loss is negative sum of radii minus penalties
        """
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Objective: maximize sum of radii -> minimize -sum(r)
        obj = -np.sum(r)
        
        # Penalties
        penalty = 0.0
        valid = True
        
        # Boundary constraints: r <= x, r <= y, r <= 1-x, r <= 1-y
        # Use soft penalty
        bnd_viol = np.maximum(0, r - np.minimum(x, np.minimum(y, np.minimum(1-x, 1-y))))
        if np.any(bnd_viol > 0):
            penalty += 10.0 * np.sum(bnd_viol**2)
            if np.any(bnd_viol > 1e-6):
                valid = False
        
        # Pairwise non-overlap: dist_ij >= r_i + r_j
        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[pair_i] + r[pair_j]
        overlap = np.maximum(0, min_dist - dist)
        if np.any(overlap > 0):
            penalty += 10.0 * np.sum(overlap**2)
            if np.any(overlap > 1e-6):
                valid = False
        
        return obj + penalty, valid
    
    def pack_loss_grad(params):
        """Gradient of pack_loss."""
        x = params[:, 0].copy()
        y = params[:, 1].copy()
        r = params[:, 2].copy()
        
        gx = np.zeros(N)
        gy = np.zeros(N)
        gr = np.zeros(N)
        
        # Objective gradient: d(-sum(r))/dr = -1
        gr[:] = -1.0
        
        # Boundary penalties
        min_bnd = np.minimum(x, np.minimum(y, np.minimum(1-x, 1-y)))
        bnd_viol = np.maximum(0, r - min_bnd)
        if np.any(bnd_viol > 0):
            # d(r - min_bnd)/dr = 1 for violating circles
            # d(r - min_bnd)/dx depends on which bound is active
            for i in range(N):
                if bnd_viol[i] > 0:
                    # Active bound
                    bounds = [x[i], y[i], 1-x[i], 1-y[i]]
                    min_b = min(bounds)
                    gr[i] += 20.0 * bnd_viol[i]  # 10 * 2 * bnd_viol
                    if bounds[0] == min_b:
                        gx[i] -= 20.0 * bnd_viol[i]
                    elif bounds[1] == min_b:
                        gy[i] -= 20.0 * bnd_viol[i]
                    elif bounds[2] == min_b:
                        gx[i] += 20.0 * bnd_viol[i]
                    elif bounds[3] == min_b:
                        gy[i] += 20.0 * bnd_viol[i]
        
        # Pairwise penalties
        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[pair_i] + r[pair_j]
        overlap = np.maximum(0, min_dist - dist)
        
        if np.any(overlap > 0):
            # For each pair with overlap
            for k in range(n_pairs):
                if overlap[k] > 0:
                    i = pair_i[k]
                    j = pair_j[k]
                    if dist[k] < 1e-10:
                        dist_k = 1e-10
                        dx_k = 0
                        dy_k = 0
                    else:
                        dist_k = dist[k]
                        dx_k = dx[k]
                        dy_k = dy[k]
                    
                    coeff = 20.0 * overlap[k]  # 10 * 2 * overlap
                    
                    # d(overlap)/dx_i = -dx/dist (since overlap = r_i+r_j-dist)
                    gx[i] += coeff * (dx_k / dist_k)
                    gx[j] -= coeff * (dx_k / dist_k)
                    gy[i] += coeff * (dy_k / dist_k)
                    gy[j] -= coeff * (dy_k / dist_k)
                    
                    # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
                    gr[i] += coeff
                    gr[j] += coeff
        
        grad = np.column_stack([gx, gy, gr])
        return grad
    
    def optimize_from(start, max_iter=500, ftol=1e-12):
        """Run L-BFGS-B optimization from a starting point."""
        from scipy.optimize import minimize
        
        params = start.copy()
        
        # Bounds: x,y in [0,1], r in [0, 0.5]
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        result = minimize(
            pack_loss,
            params,
            jac=pack_loss_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8}
        )
        
        return result.x, result.fun
    
    def make_grid_initial():
        """Create a grid-based initial guess."""
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([cx, cy, r])
        return np.array(cells[:N])
    
    def make_random_initial(rng):
        """Create a random initial guess with small circles."""
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        r = rng.uniform(0.01, 0.05, N)
        return np.column_stack([x, y, r])
    
    def make_hex_initial():
        """Create a hexagonal packing initial guess."""
        # For 32 circles, use a hexagonal arrangement
        # Approximate: rows with alternating offsets
        rows = 6
        circles = []
        # Try to fit hexagonal pattern
        r_base = 1.0 / (2 * np.sqrt(3) * rows * 0.6)  # rough estimate
        for row in range(rows):
            n_in_row = 6 if row % 2 == 0 else 5
            for col in range(n_in_row):
                if len(circles) >= N:
                    break
                x = (col + 0.5 + (0.5 if row % 2 else 0)) / (n_in_row + 0.5)
                y = (row + 0.5) / rows
                circles.append([x, y, r_base])
        # Pad if needed
        while len(circles) < N:
            circles.append([0.5, 0.5, 0.01])
        return np.array(circles[:N])
    
    def make_corner_initial():
        """Initial guess with circles in corners and center."""
        circles = []
        # 4 corners
        corner_r = 0.15
        for cx, cy in [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]:
            circles.append([cx, cy, corner_r])
        # 4 edges midpoints
        edge_r = 0.12
        for cx, cy in [(0.5, 0.12), (0.5, 0.88), (0.12, 0.5), (0.88, 0.5)]:
            circles.append([cx, cy, edge_r])
        # Center
        circles.append([0.5, 0.5, 0.18])
        # Fill remaining with smaller circles
        rng = np.random.RandomState(42)
        for _ in range(N - len(circles)):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            r = 0.05
            circles.append([x, y, r])
        return np.array(circles[:N])
    
    # Generate diverse initial guesses
    rng = np.random.RandomState(12345)
    
    initial_guesses = [
        make_grid_initial(),
        make_hex_initial(),
        make_corner_initial(),
    ]
    
    # Add random initial guesses
    for _ in range(5):
        initial_guesses.append(make_random_initial(rng))
    
    best_params = None
    best_score = -np.inf
    
    for idx, init in enumerate(initial_guesses):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        try:
            result_params, result_fun = optimize_from(init, max_iter=300)
            
            # Check validity
            _, valid = pack_loss(result_params)
            
            if valid:
                score = np.sum(result_params[:, 2])
                if score > best_score:
                    best_score = score
                    best_params = result_params.copy()
        except Exception:
            continue
    
    # If we have time, do a polish pass with tighter tolerances
    if best_params is not None and time.time() - start_time < TIME_LIMIT - 20:
        try:
            polished_params, _ = optimize_from(best_params, max_iter=500, ftol=1e-14)
            _, valid = pack_loss(polished_params)
            if valid:
                score = np.sum(polished_params[:, 2])
                if score > best_score:
                    best_score = score
                    best_params = polished_params.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to grid
        return make_grid_initial()
    
    # Final validation and slight shrink to ensure validity
    x = best_params[:, 0].copy()
    y = best_params[:, 1].copy()
    r = best_params[:, 2].copy()
    
    # Shrink radii slightly to ensure no overlap
    for _ in range(3):
        # Check all pairs
        max_overlap = 0
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j]:
                    overlap = (r[i] + r[j] - d) / 2
                    max_overlap = max(max_overlap, overlap)
        
        if max_overlap < 1e-8:
            break
        r -= max_overlap * 0.5 + 1e-6
    
    # Ensure boundary constraints
    for i in range(N):
        r[i] = min(r[i], x[i], y[i], 1-x[i], 1-y[i])
        r[i] = max(r[i], 1e-6)
    
    return np.column_stack([x, y, r])
