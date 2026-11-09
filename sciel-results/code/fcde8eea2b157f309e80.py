"""Circle packing in perimeter-4 rectangle.
Maximize sum of 21 disjoint circle radii inside a rectangle with perimeter <= 4.

Strategy:
1. Generate multiple initial layouts (grid, hexagonal, mixed-row, random).
2. For each layout, determine the optimal aspect ratio (w+h=2) and uniform radius.
3. Use a smooth surrogate objective with log-sum-exp penalties for constraints.
4. Apply multi-stage local optimization:
   a. Coarse Nelder-Mead for global-ish movement
   b. Fine BFGS/L-BFGS-B for precise convergence
   c. Final high-precision polish with tight tolerances
5. Return the layout with the maximum sum of radii.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 21


def _validate(x, y, r, w, h, tol=1e-9):
    """Check if circles are valid (inside rectangle and disjoint)."""
    n = len(x)
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < 2.0 * r - tol:
                return False
    return True


def _smooth_min_penalty(params, w, h, beta=50.0):
    """
    Smooth surrogate for the constraint violation.
    params = [r, x_0, y_0, x_1, y_1, ..., x_{n-1}, y_{n-1}]
    Uses log-sum-exp for smooth max.
    """
    n = N
    r = params[0]
    if r <= 0:
        return 1e10
    xs = params[1::2]
    ys = params[2::2]
    
    # Boundary violations (smooth max of violations)
    # Each circle must satisfy: r <= x_i <= w - r, r <= y_i <= h - r
    # Violation if x_i < r or x_i > w - r, etc.
    
    # Left wall violations: max(0, r - x_i)
    left_viol = np.maximum(0, r - xs)
    # Right wall violations: max(0, x_i - (w - r))
    right_viol = np.maximum(0, xs - (w - r))
    # Bottom wall violations: max(0, r - y_i)
    bottom_viol = np.maximum(0, r - ys)
    # Top wall violations: max(0, y_i - (h - r))
    top_viol = np.maximum(0, ys - (h - r))
    
    # Smooth max using log-sum-exp
    def smooth_max(violations, beta):
        if np.all(violations <= 0):
            return 0.0
        pos_viol = violations[violations > 0]
        if len(pos_viol) == 0:
            return 0.0
        m = np.max(pos_viol)
        return m + np.log(np.sum(np.exp((pos_viol - m) * beta))) / beta
    
    boundary_penalty = (smooth_max(left_viol, beta) + 
                        smooth_max(right_viol, beta) + 
                        smooth_max(bottom_viol, beta) + 
                        smooth_max(top_viol, beta))
    
    # Pairwise disjointness violations: max(0, 2r - dist_ij)
    pair_violations = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            viol = max(0, 2.0 * r - dist)
            if viol > 0:
                pair_violations.append(viol)
    
    if pair_violations:
        pair_violations = np.array(pair_violations)
        pair_penalty = smooth_max(pair_violations, beta)
    else:
        pair_penalty = 0.0
    
    total_penalty = boundary_penalty + pair_penalty
    return -r + 100.0 * total_penalty


def _relax(params, w, h, lr=0.01, n_iters=100):
    """Simple gradient descent with relaxation for feasibility."""
    n = N
    best_params = params.copy()
    best_obj = _smooth_min_penalty(params, w, h, beta=30.0)
    
    for it in range(n_iters):
        # Numerical gradient
        grad = np.zeros_like(params)
        eps = 1e-7
        for k in range(len(params)):
            old = params[k]
            params[k] = old + eps
            f_plus = _smooth_min_penalty(params, w, h, beta=30.0)
            params[k] = old
            grad[k] = f_plus - best_obj if k > 0 else f_plus - _smooth_min_penalty(params, w, h, beta=30.0)
            # Recompute properly
        # Use central difference for better accuracy
        grad = np.zeros_like(params)
        for k in range(len(params)):
            old = params[k]
            params[k] = old + eps
            f_plus = _smooth_min_penalty(params, w, h, beta=30.0)
            params[k] = old - eps
            f_minus = _smooth_min_penalty(params, w, h, beta=30.0)
            params[k] = old
            grad[k] = (f_plus - f_minus) / (2 * eps)
        
        params = params - lr * grad
        # Ensure r > 0
        params[0] = max(params[0], 1e-6)
        
        obj = _smooth_min_penalty(params, w, h, beta=30.0)
        if obj < best_obj:
            best_obj = obj
            best_params = params.copy()
    
    return best_params


def _scipy_polish(params0, w, h, max_iters=2000):
    """High-precision polish using L-BFGS-B."""
    try:
        result = minimize(_smooth_min_penalty, params0, args=(w, h), 
                         method='L-BFGS-B',
                         options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10})
        return result.x
    except Exception:
        return params0


def _grid_layout(n_cols, n_rows):
    """Generate grid layout."""
    xs = []
    ys = []
    for row in range(n_rows):
        for col in range(n_cols):
            xs.append(col * 2.0 + 1.0)
            ys.append(row * 2.0 + 1.0)
    xs = np.array(xs)
    ys = np.array(ys)
    W = (n_cols - 1) * 2.0 + 2.0
    H = (n_rows - 1) * 2.0 + 2.0
    return xs, ys, W, H


def _hex_layout(n_cols, n_rows):
    """Generate hexagonal layout."""
    xs = []
    ys = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * 2.0 + (1.0 if row % 2 == 1 else 0.0)
            y = row * np.sqrt(3.0)
            xs.append(x)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - xs.min() + 1.0
    ys = ys - ys.min() + 1.0
    W = xs.max() - xs.min() + 2.0
    H = ys.max() - ys.min() + 2.0
    return xs, ys, W, H


def _mixed_row_layout(row_counts):
    """Generate staggered layout with varying row counts."""
    xs = []
    ys = []
    for row_idx, n_circ in enumerate(row_counts):
        y = row_idx * np.sqrt(3.0)
        offset = 1.0 if row_idx % 2 == 1 else 0.0
        for col in range(n_circ):
            x = col * 2.0 + offset
            xs.append(x)
            ys.append(y)
    if len(xs) == 0:
        return None
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - xs.min() + 1.0
    ys = ys - ys.min() + 1.0
    W = xs.max() - xs.min() + 2.0
    H = ys.max() - ys.min() + 2.0
    return xs, ys, W, H


def _random_layout(n, seed=42):
    """Generate a random layout."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0, 1, n)
    ys = rng.uniform(0, 1, n)
    xs = xs - xs.min() + 1.0
    ys = ys - ys.min() + 1.0
    W = xs.max() - xs.min() + 2.0
    H = ys.max() - ys.min() + 2.0
    return xs, ys, W, H


def _compute_initial_layout(xs, ys, n_circles):
    """Compute initial layout with optimal aspect ratio."""
    if len(xs) < n_circles:
        return None
    
    xs_sub = xs[:n_circles]
    ys_sub = ys[:n_circles]
    
    W_sub = xs_sub.max() - xs_sub.min() + 2.0
    H_sub = ys_sub.max() - ys_sub.min() + 2.0
    
    r = 2.0 / (W_sub + H_sub)
    w = 2.0 * W_sub / (W_sub + H_sub)
    h = 2.0 * H_sub / (W_sub + H_sub)
    
    x0 = (xs_sub - xs_sub.min()) * r + r
    y0 = (ys_sub - ys_sub.min()) * r + r
    
    return x0, y0, w, h, r


def _optimize_single(x0, y0, w, h, r0, deadline):
    """Multi-stage optimization for a single layout."""
    n = N
    params0 = np.concatenate([[r0], x0, y0])
    
    # Stage 1: Coarse Nelder-Mead
    best_params = params0.copy()
    best_obj = _smooth_min_penalty(params0, w, h, beta=20.0)
    
    try:
        result1 = minimize(_smooth_min_penalty, params0, args=(w, h),
                          method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if result1.fun < best_obj:
            best_params = result1.x
            best_obj = result1.fun
    except Exception:
        pass
    
    # Stage 2: Relaxation for feasibility
    if time.time() < deadline - 20:
        relaxed = _relax(best_params, w, h, lr=0.005, n_iters=50)
        obj_relaxed = _smooth_min_penalty(relaxed, w, h, beta=30.0)
        if obj_relaxed < best_obj:
            best_params = relaxed
            best_obj = obj_relaxed
    
    # Stage 3: High-precision polish
    if time.time() < deadline - 10:
        polished = _scipy_polish(best_params, w, h, max_iters=1000)
        obj_polished = _smooth_min_penalty(polished, w, h, beta=50.0)
        if obj_polished < best_obj:
            best_params = polished
            best_obj = obj_polished
    
    # Extract result
    r_final = max(best_params[0], 1e-6)
    x_final = best_params[1::2]
    y_final = best_params[2::2]
    
    # Validate
    if not _validate(x_final, y_final, r_final, w, h, tol=1e-8):
        # Try to repair
        r_final = min(r_final, np.min(np.minimum(x_final, w - x_final, y_final, h - y_final)))
        if not _validate(x_final, y_final, r_final, w, h, tol=1e-8):
            return None
    
    return np.column_stack([x_final, y_final, r_final]), n * r_final


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_sum = 0.0
    best_result = None
    
    # Collect candidate layouts
    candidates = []
    
    # Grid layouts
    for nc in range(1, 12):
        for nr in range(1, 12):
            if nc * nr >= N:
                candidates.append(('grid', nc, nr))
    
    # Hexagonal layouts
    for nc in range(1, 12):
        for nr in range(1, 12):
            if nc * nr >= N:
                candidates.append(('hex', nc, nr))
    
    # Mixed row layouts
    base_counts = [
        [4, 4, 4, 4, 3, 2],
        [5, 4, 4, 4, 3, 1],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [5, 5, 4, 3, 2, 2],
        [6, 4, 4, 3, 2, 2],
        [4, 5, 4, 4, 3, 1],
        [3, 4, 4, 5, 3, 2],
        [5, 4, 5, 3, 2, 2],
        [4, 4, 5, 4, 3, 1],
        [7, 5, 4, 3, 2],
        [6, 6, 4, 3, 2],
        [5, 5, 5, 4, 2],
        [4, 4, 4, 4, 4, 1],
        [6, 5, 4, 4, 2],
        [5, 5, 4, 4, 3],
        [7, 4, 4, 4, 2],
        [6, 4, 5, 4, 2],
        [5, 6, 4, 4, 2],
        [4, 5, 5, 4, 3],
        [3, 3, 4, 4, 4, 3],
        [2, 4, 4, 4, 4, 3],
        [4, 4, 4, 3, 4, 2],
        [5, 3, 4, 4, 3, 2],
    ]
    for counts in base_counts:
        if sum(counts) >= N:
            candidates.append(('mixed', counts))
    
    # Random layouts
    for seed in range(30):
        candidates.append(('random', seed))
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline - 30.0:
            break
        
        try:
            if cand[0] == 'grid':
                _, nc, nr = cand
                xs, ys, W, H = _grid_layout(nc, nr)
            elif cand[0] == 'hex':
                _, nc, nr = cand
                xs, ys, W, H = _hex_layout(nc, nr)
            elif cand[0] == 'mixed':
                _, counts = cand
                res = _mixed_row_layout(counts)
                if res is None:
                    continue
                xs, ys, W, H = res
            elif cand[0] == 'random':
                _, seed = cand
                xs, ys, W, H = _random_layout(N, seed)
            else:
                continue
            
            init = _compute_initial_layout(xs, ys, N)
            if init is None:
                continue
            
            x0, y0, w, h, r0 = init
            
            # Quick check of initial sum
            init_sum = N * r0
            
            # Only optimize if promising
            if init_sum > best_sum * 0.95 or best_sum == 0:
                result, opt_sum = _optimize_single(x0, y0, w, h, r0, deadline)
                if result is not None and opt_sum > best_sum:
                    best_sum = opt_sum
                    best_result = result
                    
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
