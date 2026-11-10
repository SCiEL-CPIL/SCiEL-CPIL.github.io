# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns base positions for 64 tubes.
    """
    N = 8
    start_time = time.time()
    time_limit = 600.0
    margin = 60.0
    
    # 1D Union Length Function
    # x: array of 8 base positions
    # c: scalar height
    # Intervals: [x[i] + c*i/N, x[i] + 1/N + c*i/N]
    def union_length_1d(x, c):
        # Create intervals
        i = np.arange(N)
        starts = x + c * i / N
        ends = starts + 1.0 / N
        
        # Sort by start
        idx = np.argsort(starts)
        starts = starts[idx]
        ends = ends[idx]
        
        # Compute union length using sweep line
        length = 0.0
        curr_start = starts[0]
        curr_end = ends[0]
        for i in range(1, N):
            if starts[i] <= curr_end:
                curr_end = max(curr_end, ends[i])
            else:
                length += curr_end - curr_start
                curr_start = starts[i]
                curr_end = ends[i]
        length += curr_end - curr_start
        return length

    # Precompute the slopes for each interval endpoint
    # Interval i has slope i/N for both start and end
    slopes = np.arange(N, dtype=np.float64) / N
    # Each interval contributes 2 endpoints (start and end), both with slope i/N
    # So we have 2N = 16 endpoints total
    # Endpoint indices: 0..7 are starts (slope = i/N), 8..15 are ends (slope = i/N)
    
    def objective_1d_exact(x):
        """
        Compute the exact integral of union_length_1d(x, c) dc from 0 to 1.
        Uses vectorized breakpoint detection.
        """
        # Endpoint intercepts: x[i] for starts, x[i] + 1/N for ends
        intercepts = np.empty(2 * N, dtype=np.float64)
        intercepts[:N] = x
        intercepts[N:] = x + 1.0 / N
        
        # Endpoint slopes: i/N for both start and end of interval i
        ep_slopes = np.concatenate([slopes, slopes])
        
        # Find all c values where two endpoints are equal
        # a1 + b1*c = a2 + b2*c => c = (a2-a1)/(b1-b2) if b1 != b2
        # Vectorized: compute all pairwise differences
        # For 16 endpoints, that's 16*15/2 = 120 pairs
        
        # Build upper triangle indices
        i_idx, j_idx = np.triu_indices(2 * N, k=1)
        da = intercepts[j_idx] - intercepts[i_idx]
        db = ep_slopes[i_idx] - ep_slopes[j_idx]
        
        # c_val = da / db where db != 0
        valid = np.abs(db) > 1e-12
        c_vals = np.zeros(np.sum(valid), dtype=np.float64)
        c_vals[:] = da[valid] / db[valid]
        
        # Filter to [0, 1]
        c_vals = c_vals[(c_vals >= 0.0) & (c_vals <= 1.0)]
        
        # Add boundaries and sort
        breakpoints = np.concatenate([[0.0], c_vals, [1.0]])
        breakpoints = np.unique(np.sort(breakpoints))
        
        # For each interval between consecutive breakpoints, union length is linear in c
        # Use trapezoidal rule (exact for linear functions)
        n_bp = len(breakpoints)
        if n_bp < 2:
            return 0.0
        
        # Evaluate union length at all breakpoints
        lens = np.empty(n_bp, dtype=np.float64)
        for k in range(n_bp):
            lens[k] = union_length_1d(x, breakpoints[k])
        
        # Trapezoidal integration
        dc = np.diff(breakpoints)
        total = np.sum((lens[:-1] + lens[1:]) / 2.0 * dc)
        
        return total

    # Optimization
    # Use Nelder-Mead with exact objective and multi-restart strategy
    
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Generate multiple initial points for multi-restart
    rng_init = np.random.default_rng(42)
    initial_points = [keich.copy()]
    
    # Add perturbed versions of Keich
    for _ in range(5):
        perturbed = keich + rng_init.normal(0, 0.05, size=N)
        initial_points.append(perturbed)
    
    # Add some random initial points
    for _ in range(3):
        random_x = rng_init.uniform(-0.5, 1.5, size=N)
        initial_points.append(random_x)
    
    # Multi-restart optimization
    best_x = None
    best_val = np.inf
    
    for x0 in initial_points:
        if time.time() - start_time > time_limit - margin:
            break
        
        res = opt.minimize(objective_1d_exact, x0, method='Nelder-Mead',
                           options={'maxiter': 15000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        val = res.fun
        if val < best_val:
            best_val = val
            best_x = res.x
    
    x_opt = best_x if best_x is not None else keich.copy()
    
    # Check time
    if time.time() - start_time > time_limit - margin:
        y_opt = x_opt.copy()
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = x_opt[i_flat]
        base_y = y_opt[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        return positions
    
    # Polish pass with tighter tolerance
    res2 = opt.minimize(objective_1d_exact, x_opt, method='Nelder-Mead',
                        options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    
    x_opt = res2.x
    
    # Check time again
    if time.time() - start_time > time_limit - margin:
        y_opt = x_opt.copy()
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = x_opt[i_flat]
        base_y = y_opt[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        return positions
    
    # Try asymmetric x and y
    # Optimize y separately
    def objective_1d_y(y):
        return objective_1d_exact(y)
    
    y0 = x_opt.copy()
    res_y = opt.minimize(objective_1d_y, y0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
    
    y_opt = res_y.x
    
    # Check time
    if time.time() - start_time > time_limit - margin:
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = x_opt[i_flat]
        base_y = y_opt[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        return positions
    
    # Final polish on both x and y with very tight tolerance
    if time.time() - start_time < time_limit - margin - 30:
        res_x_final = opt.minimize(objective_1d_exact, x_opt, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        x_opt = res_x_final.x
        
        if time.time() - start_time < time_limit - margin - 15:
            res_y_final = opt.minimize(objective_1d_y, y_opt, method='Nelder-Mead',
                                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            y_opt = res_y_final.x
    
    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Sanity check: ensure finite and valid
    if not np.all(np.isfinite(positions)):
        # Fallback to simple grid if something went wrong
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
