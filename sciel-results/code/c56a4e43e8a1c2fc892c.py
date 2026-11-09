"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _validity_check(x, y, r, tol=1e-9):
    """Returns True if configuration is valid, False otherwise"""
    n = len(r)
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
        return False
    # Vectorized overlap check
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx*dx + dy*dy)
    if np.any(dist < r[i_idx] + r[j_idx] - tol):
        return False
    return True

def _make_valid(x, y, r, tol=1e-9):
    """Ensure configuration is valid by shrinking radii if needed"""
    r = r.copy()
    # First ensure bounds
    for i in range(len(r)):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        if r[i] > max_r:
            r[i] = max(0, max_r)
    
    i_idx, j_idx = np.triu_indices(len(r), k=1)
    
    # Iteratively fix overlaps
    for _ in range(500):
        # Vectorized overlap calculation
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        overlaps = r[i_idx] + r[j_idx] - dist
        
        max_overlap = np.max(overlaps)
        if max_overlap <= tol:
            break
        
        # Find the pair with max overlap
        max_pair_idx = np.argmax(overlaps)
        max_i = i_idx[max_pair_idx]
        max_j = j_idx[max_pair_idx]
        
        # Shrink the larger circle by half the overlap plus a small buffer
        shrink = max_overlap / 2 + 1e-7
        if r[max_i] >= r[max_j]:
            r[max_i] = max(0, r[max_i] - shrink)
        else:
            r[max_j] = max(0, r[max_j] - shrink)
    
    # Final safety: if still invalid, scale down all radii
    if not _validity_check(x, y, r, tol):
        r *= 0.95
        for _ in range(50):
            if _validity_check(x, y, r, tol):
                break
            r *= 0.9
        
    return x, y, r

def _maximize_radii(x, y, r, iterations=50, tol=1e-10):
    """Try to increase each radius to its maximum possible value"""
    r = r.copy()
    for _ in range(iterations):
        changed = False
        for i in range(len(r)):
            # Max radius from boundary
            r_bound = min(x[i], 1 - x[i], y[i], 1 - y[i])
            # Max radius from neighbors
            r_neighbor = 0.5
            for j in range(len(r)):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                r_neighbor = min(r_neighbor, max(0, dist - r[j] - tol))
            
            new_r = min(r_bound, r_neighbor)
            if new_r > r[i] + 1e-12:
                r[i] = new_r
                changed = True
        
        if not changed:
            break
    
    # Ensure validity after maximization
    if not _validity_check(x, y, r, 1e-9):
        x, y, r = _make_valid(x, y, r)
    
    return x, y, r

def _hexagonal_lattice(n, margin=0.05):
    """
    Generate n points on a hexagonal lattice inside [margin, 1-margin]^2.
    Returns array of shape (n, 2).
    """
    points = []
    # For a hex lattice, we want circles of radius r to be tangent.
    # Distance between centers in hex lattice = 2r (for tangent circles)
    # Horizontal spacing = 2r, vertical spacing = 2r * sqrt(3)/2 = r*sqrt(3)
    # Area per circle in hex packing = 2r^2 * sqrt(3)
    # Total area = (1-2*margin)^2
    # So 2r^2 * sqrt(3) * n <= (1-2*margin)^2
    # r <= sqrt((1-2*margin)^2 / (2 * sqrt(3) * n))
    
    side = 1.0 - 2.0 * margin
    r_est = np.sqrt(side * side / (2.0 * np.sqrt(3) * n))
    
    # Spacing between centers
    col_spacing = 2.0 * r_est
    row_spacing = r_est * np.sqrt(3)
    
    # Determine number of rows and columns
    n_rows = int(np.floor(side / row_spacing)) + 1
    n_cols = int(np.floor(side / col_spacing)) + 1
    
    # Generate points
    for i in range(n_rows):
        y = margin + i * row_spacing
        offset = (i % 2) * (col_spacing / 2.0)
        for j in range(n_cols):
            x = margin + j * col_spacing + offset
            if margin <= x <= 1.0 - margin and margin <= y <= 1.0 - margin:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than n points, select the best n (most central)
    if len(points) > n:
        # Prefer points closer to center
        center = 0.5
        dists = np.sqrt((points[:, 0] - center)**2 + (points[:, 1] - center)**2)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        # Pad with random points
        while len(points) < n:
            points = np.vstack([points, np.array([np.random.uniform(margin, 1-margin), 
                                                   np.random.uniform(margin, 1-margin)])])
    
    return points[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_result = None
    best_score = -np.inf
    
    # Precompute indices for vectorized overlap calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def objective(params):
        """
        Objective function for L-BFGS-B.
        We want to maximize sum(r), so we minimize -sum(r).
        We add smooth penalties for boundary violations and circle overlaps.
        Vectorized for speed.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Ensure radii are positive
        r_pos = np.maximum(r, 1e-6)
        
        # 1. Boundary Penalties (smooth)
        # Violation: r - x < 0 means circle extends left
        viol_x_left = r_pos - x
        viol_x_right = x + r_pos - 1.0
        viol_y_bottom = r_pos - y
        viol_y_top = y + r_pos - 1.0
        
        # Smooth penalty using soft-max approximation
        beta = 20.0
        pen_x_left = np.logaddexp(0, beta * viol_x_left) / beta
        pen_x_right = np.logaddexp(0, beta * viol_x_right) / beta
        pen_y_bottom = np.logaddexp(0, beta * viol_y_bottom) / beta
        pen_y_top = np.logaddexp(0, beta * viol_y_top) / beta
        boundary_pen = np.sum(pen_x_left + pen_x_right + pen_y_bottom + pen_y_top)
        
        # 2. Overlap Penalties (Vectorized, smooth)
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        d_sq = dx*dx + dy*dy
        d = np.sqrt(d_sq + 1e-12)  # Avoid division by zero
        overlaps = r_pos[i_idx] + r_pos[j_idx] - d
        # Smooth penalty
        overlap_pen = np.sum(np.logaddexp(0, beta * overlaps) / beta)
        
        # Weights: High weight to ensure validity is prioritized
        w_pen = 50.0
        
        return -np.sum(r_pos) + w_pen * (boundary_pen + overlap_pen)
    
    def optimize_start(x0, y0, r0, maxiter=500, ftol=1e-12, gtol=1e-8):
        """
        Run L-BFGS-B optimization from a given start.
        Returns the best valid configuration found.
        """
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
        for i in range(N):
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0, 0.5))  # r
            
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Try to repair by making valid
            x_opt, y_opt, r_opt = _make_valid(x_opt, y_opt, r_opt)
            
            # Try to maximize radii further
            x_opt, y_opt, r_opt = _maximize_radii(x_opt, y_opt, r_opt)
            
            score = np.sum(r_opt)
            
            return score, np.column_stack((x_opt, y_opt, r_opt))
            
        except Exception:
            return -np.inf, None

    def polish(x, y, r):
        """
        High-precision polish pass.
        Runs multiple rounds of L-BFGS-B with progressively tighter tolerances.
        """
        best_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Multiple polish stages with decreasing tolerance
        stages = [
            {'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10},
            {'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-11},
            {'maxiter': 4000, 'ftol': 1e-16, 'gtol': 1e-12},
        ]
        
        for stage in stages:
            if time.time() > deadline - 5:
                break
            
            params0 = np.concatenate([best_x, best_y, best_r])
            bounds = []
            for i in range(N):
                bounds.append((0, 1))  # x
            for i in range(N):
                bounds.append((0, 1))  # y
            for i in range(N):
                bounds.append((0, 0.5))  # r
                
            try:
                res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                               options=stage)
                
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                
                # Repair and maximize
                x_opt, y_opt, r_opt = _make_valid(x_opt, y_opt, r_opt)
                x_opt, y_opt, r_opt = _maximize_radii(x_opt, y_opt, r_opt, iterations=100)
                
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x_opt, y_opt, r_opt
                    
            except Exception:
                pass
        
        return best_score, np.column_stack((best_x, best_y, best_r))
    
    def local_search(x, y, r, num_iter=15):
        """
        Simple local search: perturb positions/radii, re-optimize.
        """
        best_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for it in range(num_iter):
            if time.time() > deadline - 10:
                break
            
            # Adaptive perturbation scale
            perturb_scale = 0.03 * (1 - it / num_iter) + 0.01
            
            # Perturb
            x_new = best_x + np.random.normal(0, perturb_scale, N)
            y_new = best_y + np.random.normal(0, perturb_scale, N)
            r_new = best_r + np.random.normal(0, 0.005, N)
            
            # Keep within bounds
            x_new = np.clip(x_new, 0, 1)
            y_new = np.clip(y_new, 0, 1)
            r_new = np.clip(r_new, 0, 0.5)
            
            # Optimize from perturbed state with fewer iterations
            score, config = optimize_start(x_new, y_new, r_new, maxiter=300)
            
            if config is not None and score > best_score:
                best_score = score
                best_x, best_y, best_r = config[:, 0], config[:, 1], config[:, 2]
        
        return best_score, np.column_stack((best_x, best_y, best_r))
    
    # Try multiple random restarts with different initializations
    n_restarts = 12
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Initialize with hexagonal lattice + random perturbation
        x0 = _hexagonal_lattice(N, margin=0.05)
        # Add small random perturbation
        x0 += rng.uniform(-0.01, 0.01, size=x0.shape)
        
        # Initial radii: estimate from hex packing
        # In hex packing, r = spacing / 2
        # spacing ≈ sqrt(area / (n * sqrt(3)/2))
        r_est = np.sqrt(1.0 / (N * np.sqrt(3) / 2)) * 0.5
        r0 = np.full(N, max(0.02, r_est * 0.8))
        
        # Optimize
        score, config = optimize_start(x0[:, 0], x0[:, 1], r0)
        
        if config is not None and score > best_score:
            best_score = score
            best_result = config
        
        # If we have time, do local search and polish
        if time.time() < deadline - 25.0 and best_result is not None:
            x_cur = best_result[:, 0]
            y_cur = best_result[:, 1]
            r_cur = best_result[:, 2]
            
            # Local search
            ls_score, ls_config = local_search(x_cur, y_cur, r_cur, num_iter=8)
            
            if ls_score > best_score:
                best_score = ls_score
                best_result = ls_config
            
            # Polish
            if time.time() < deadline - 20.0:
                pol_score, pol_config = polish(best_result[:, 0], best_result[:, 1], best_result[:, 2])
                
                if pol_score > best_score:
                    best_score = pol_score
                    best_result = pol_config
    
    # If no valid result found, return a trivial valid packing
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
