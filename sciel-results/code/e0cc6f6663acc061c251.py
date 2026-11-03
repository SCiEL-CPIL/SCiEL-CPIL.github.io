import numpy as np
import time
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 140  # seconds, leave margin for 150s kill

def _check_validity(x, y, r, w, h):
    """Check if configuration is valid (no overlaps, within bounds)."""
    for i in range(N):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9:
            return False
        if y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - 1e-9:
                return False
    return True

def _total_radius(x, y, r):
    return np.sum(r)

def _grid_init(w, h):
    """Generate a grid-like initialization for N circles in w x h rectangle."""
    # Use a 5x5 grid and pick 21 of the 25 cells
    side = 5
    xs = []
    ys = []
    for i in range(side):
        for j in range(side):
            xs.append((i + 0.5) / side * w)
            ys.append((j + 0.5) / side * h)
    xs = np.array(xs[:N])
    ys = np.array(ys[:N])
    # Initial radii: conservative
    max_r = min(w, h) / (2 * side) * 0.95
    r = np.full(N, max_r)
    return xs, ys, r

def _random_init(w, h):
    """Random initialization."""
    xs = np.random.uniform(0.05, w - 0.05, N)
    ys = np.random.uniform(0.05, h - 0.05, N)
    r = np.full(N, min(w, h) / (2 * 5) * 0.8)
    return xs, ys, r

def _local_optimize(x0, y0, r0, w, h, time_budget):
    """
    Local optimization using Nelder-Mead on (x, y, r) with penalty method.
    Maximizes sum of radii subject to non-overlap and boundary constraints.
    """
    start = time.time()
    
    def objective(vars):
        x = vars[0:N]
        y = vars[N:2*N]
        r = np.maximum(vars[2*N:3*N], 1e-6)
        
        sum_r = np.sum(r)
        
        # Overlap penalty
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap_pen += (min_d - d)**2
        
        # Boundary penalty
        bound_pen = 0.0
        for i in range(N):
            if x[i] < r[i]:
                bound_pen += (r[i] - x[i])**2
            if x[i] > w - r[i]:
                bound_pen += (x[i] - (w - r[i]))**2
            if y[i] < r[i]:
                bound_pen += (r[i] - y[i])**2
            if y[i] > h - r[i]:
                bound_pen += (y[i] - (h - r[i]))**2
        
        penalty = 500.0 * (overlap_pen + bound_pen)
        return -sum_r + penalty
    
    x0_flat = np.concatenate([x0, y0, r0])
    
    try:
        res = minimize(objective, x0_flat, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7,
                                'adaptive': True})
    except Exception:
        return x0, y0, r0, _total_radius(x0, y0, r0)
    
    x_opt = res.x[0:N]
    y_opt = res.x[N:2*N]
    r_opt = np.maximum(res.x[2*N:3*N], 1e-6)
    
    # Post-process: shrink radii to ensure validity
    # Check boundary
    for i in range(N):
        r_opt[i] = min(r_opt[i], x_opt[i], w - x_opt[i], y_opt[i], h - y_opt[i])
        r_opt[i] = max(r_opt[i], 1e-6)
    
    # Check overlaps and shrink
    for _ in range(10):
        max_viol = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2)
                min_d = r_opt[i] + r_opt[j]
                if d < min_d:
                    viol = (min_d - d) / 2
                    max_viol = max(max_viol, viol)
        if max_viol <= 1e-9:
            break
        r_opt = np.maximum(r_opt - max_viol - 1e-6, 1e-6)
    
    score = _total_radius(x_opt, y_opt, r_opt)
    return x_opt, y_opt, r_opt, score

def _successive_relaxation(x, y, r, w, h, max_iters=50):
    """
    Successive convex relaxation: iteratively solve a QP-like subproblem
    by linearizing the non-overlap constraints and updating radii.
    """
    start = time.time()
    for it in range(max_iters):
        if time.time() - start > 10:
            break
        
        # Build active constraints
        # For each pair, check if they are close (active)
        # Solve for new radii given fixed positions
        # Simple approach: for each circle, its max radius is limited by
        # nearest neighbor and boundaries
        
        new_r = np.zeros(N)
        for i in range(N):
            # Boundary limit
            max_r_i = min(x[i], w - x[i], y[i], h - y[i])
            
            # Neighbor limits
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                # r_i + r_j <= d => r_i <= d - r_j
                # We want to maximize sum, so set r_i as large as possible
                # but this is coupled. Use iterative update.
                max_r_i = min(max_r_i, d - r[j] + 1e-9)
            
            new_r[i] = max(max_r_i, 1e-6)
        
        # Update radii with damping
        r = 0.5 * r + 0.5 * new_r
        
        # Check if converged
        if np.max(np.abs(new_r - r)) < 1e-8:
            break
    
    return r

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_config = None
    
    # Try different aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h = 2
    # Try w from 0.5 to 1.5 (h = 2 - w)
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15, 0.75, 1.25]
    
    for w in aspect_ratios:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        h = 2.0 - w
        
        # Try multiple initializations
        inits = []
        # Grid init
        xs, ys, r = _grid_init(w, h)
        inits.append((xs, ys, r))
        
        # Random inits
        for _ in range(3):
            xs, ys, r = _random_init(w, h)
            inits.append((xs, ys, r))
        
        for xs, ys, r in inits:
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Local optimization
            x_opt, y_opt, r_opt, score = _local_optimize(xs, ys, r, w, h, 15)
            
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x_opt, y_opt, r_opt))
    
    # Final polish: try to improve best config with more iterations
    if best_config is not None and time.time() - start_time < TIME_LIMIT - 20:
        x = best_config[:, 0]
        y = best_config[:, 1]
        r = best_config[:, 2]
        
        # Determine w, h from best config
        # We need to track which aspect ratio gave the best
        # For now, re-optimize with the same w, h
        # Actually, let's just do a few more Nelder-Mead iterations on the best
        
        # Find which aspect ratio was used
        # We'll just try to polish with a generic approach
        for w_test in aspect_ratios:
            h_test = 2.0 - w_test
            # Check if best_config fits in this rectangle
            if np.max(x) <= w_test + 1e-6 and np.max(y) <= h_test + 1e-6:
                x_opt, y_opt, r_opt, score = _local_optimize(x, y, r, w_test, h_test, 20)
                if score > best_score:
                    best_score = score
                    best_config = np.column_stack((x_opt, y_opt, r_opt))
                break
    
    if best_config is None:
        # Fallback
        w, h = 1.0, 1.0
        xs, ys, r = _grid_init(w, h)
        return np.column_stack((xs, ys, r))
    
    return best_config
