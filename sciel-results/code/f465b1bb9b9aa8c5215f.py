import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
TIME_LIMIT = 145.0  # Safety margin

def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns array of shape (C(n,3),)."""
    n = len(pts)
    # Vectorized computation
    # i < j < k
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # We can do this with broadcasting but it creates a huge array (13x13x13).
    # 13^3 is small (2197), so it's fine.
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Create index arrays
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Filter for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    xi, xj, xk = x[i_flat], x[j_flat], x[k_flat]
    yi, yj, yk = y[i_flat], y[j_flat], y[k_flat]
    
    # Cross product
    cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area_fast(pts):
    """Compute convex hull area efficiently by checking convex position first."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Centroid
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    
    # Angles
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    sorted_idx = np.argsort(angles)
    sorted_pts = pts[sorted_idx]
    
    # Compute area of polygon formed by sorted points
    x = sorted_pts[:, 0]
    y = sorted_pts[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    
    # Check if the polygon is convex
    # Edge vectors
    ex = np.roll(x, -1) - x
    ey = np.roll(y, -1) - y
    # Cross products of consecutive edges
    # e_i x e_{i+1}
    # e_i = (ex[i], ey[i]), e_{i+1} = (ex[i+1], ey[i+1])
    # Cross = ex[i]*ey[i+1] - ey[i]*ex[i+1]
    ex_next = np.roll(ex, -1)
    ey_next = np.roll(ey, -1)
    cross = ex * ey_next - ey * ex_next
    
    # For a strictly convex polygon, all cross products should have the same sign
    # (and non-zero). If any are zero or have different signs, it's not strictly convex
    # or points are collinear.
    # We consider it "convex enough" if all are positive (CCW) or all negative (CW).
    # Since we sorted by angle, it should be CCW, so cross > 0.
    is_convex = np.all(cross > 1e-9)
    
    if is_convex:
        return area
    else:
        # Fall back to scipy
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0

def _true_score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull_area = _hull_area_fast(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_all_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0

def _neg_log_soft_min_ratio(x, beta):
    """
    Objective function for minimization.
    Minimizes: - (soft_min(areas) / hull_area)
    Equivalent to minimizing: -log(soft_min(areas)) + log(hull_area)
    soft_min = (1/beta) * log(sum(exp(-beta * a)))
    -log(soft_min) = -log(sum(exp(-beta * a))) + log(beta)
    So we minimize: -log(sum(exp(-beta * a))) + log(hull_area) + log(beta)
    Since log(beta) is constant, we minimize: -log(sum(exp(-beta * a))) + log(hull_area)
    """
    pts = x.reshape(N, 2)
    
    areas = _all_triangle_areas(pts)
    if np.any(areas < 0):
        return 1e10
        
    hull_area = _hull_area_fast(pts)
    if hull_area < 1e-9:
        return 1e10
    
    # Log-sum-exp trick for stability
    # We want log(sum(exp(-beta * a)))
    # = -beta * max(a) + log(sum(exp(-beta * (a - max(a)))))
    max_a = np.max(areas)
    shifted = areas - max_a
    log_sum_exp = -beta * max_a + np.log(np.sum(np.exp(-beta * shifted)))
    
    # Objective: -log_sum_exp + log(hull_area)
    # We want to MAXIMIZE soft_min/hull_area
    # soft_min = -log_sum_exp / beta
    # Ratio = soft_min / hull_area = -log_sum_exp / (beta * hull_area)
    # Log Ratio = -log(-log_sum_exp) - log(beta) - log(hull_area)  <-- This is tricky if log_sum_exp is positive/negative.
    
    # Let's stick to the original form but optimized.
    # soft_min = -log_sum_exp / beta
    # We want to minimize -soft_min/hull_area
    # = (log_sum_exp / beta) / hull_area
    # = log_sum_exp / (beta * hull_area)
    
    return log_sum_exp / (beta * hull_area)

def _local_polish(pts, time_budget, start_time):
    """
    High-precision local search using Nelder-Mead on the true score.
    """
    deadline = start_time + time_budget
    
    # Use Nelder-Mead with tight tolerances
    def neg_score(x):
        return -_true_score(x.reshape(N, 2))
    
    best_x = pts.flatten()
    best_score = -neg_score(best_x)
    
    # Run Nelder-Mead
    try:
        res = minimize(neg_score, best_x, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 1000})
        
        if time.time() < deadline:
            if res.fun < -best_score:
                best_x = res.x
                best_score = -res.fun
    except Exception:
        pass
    
    # Final coordinate descent polish
    # Try small steps in each direction
    step_sizes = [1e-4, 1e-5, 1e-6]
    for step in step_sizes:
        if time.time() > deadline - 1:
            break
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                for d in range(2):
                    for delta in [step, -step]:
                        trial_x = best_x.copy()
                        trial_x[i*2 + d] += delta
                        trial_pts = trial_x.reshape(N, 2)
                        s = _true_score(trial_pts)
                        if s > best_score + 1e-12:
                            best_x = trial_x
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best_x.reshape(N, 2), best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    best_x = x0.copy()
    best_score = _true_score(x0.reshape(N, 2))
    
    # List of betas for soft-min annealing
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
            
        def objective(x):
            return _neg_log_soft_min_ratio(x, beta)
        
        # Run L-BFGS-B
        try:
            res = minimize(objective, current_x, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() < deadline:
                # Evaluate true score of the result
                trial_pts = res.x.reshape(N, 2)
                trial_score = _true_score(trial_pts)
                
                if trial_score > best_score:
                    best_score = trial_score
                    best_x = res.x.copy()
                    
                # Update current_x for next iteration
                # Use the result from the optimizer, even if it didn't improve the global best,
                # as it might be a better starting point for the next beta.
                current_x = res.x.copy()
                
        except Exception:
            pass
    
    # Final local polish on the best result found
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5:
        polished_pts, polished_score = _local_polish(best_x.reshape(N, 2), remaining_time, start_time)
        if polished_score > best_score:
            best_score = polished_score
            best_x = polished_pts.flatten()
    
    return best_x.reshape(N, 2)
