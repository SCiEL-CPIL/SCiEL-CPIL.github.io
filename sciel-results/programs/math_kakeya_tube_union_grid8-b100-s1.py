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

    def get_breakpoints(x):
        """
        Find all c in [0,1] where the ordering of interval endpoints changes.
        """
        # Endpoint intercepts: x[i] for starts, x[i] + 1/N for ends
        intercepts = np.empty(2 * N, dtype=np.float64)
        intercepts[:N] = x
        intercepts[N:] = x + 1.0 / N
        
        # Endpoint slopes: i/N for both start and end of interval i
        slopes = np.arange(N, dtype=np.float64) / N
        ep_slopes = np.concatenate([slopes, slopes])
        
        # Find all c values where two endpoints are equal
        i_idx, j_idx = np.triu_indices(2 * N, k=1)
        da = intercepts[j_idx] - intercepts[i_idx]
        db = ep_slopes[i_idx] - ep_slopes[j_idx]
        
        valid = np.abs(db) > 1e-12
        c_vals = da[valid] / db[valid]
        
        # Filter to [0, 1]
        c_vals = c_vals[(c_vals >= 0.0) & (c_vals <= 1.0)]
        
        # Add boundaries and sort
        breakpoints = np.concatenate([[0.0], c_vals, [1.0]])
        breakpoints = np.unique(np.sort(breakpoints))
        return breakpoints

    def coupled_volume(X, Y):
        """
        Compute the exact volume of the union of tubes.
        V = integral_0^1 L_x(c) * L_y(c) dc
        """
        bx = get_breakpoints(X)
        by = get_breakpoints(Y)
        
        # Merge breakpoints
        b = np.unique(np.concatenate([bx, by]))
        
        if len(b) < 2:
            return 0.0
            
        total_vol = 0.0
        for k in range(len(b) - 1):
            c1 = b[k]
            c2 = b[k + 1]
            if c2 <= c1:
                continue
                
            # Evaluate lengths at endpoints
            lx1 = union_length_1d(X, c1)
            lx2 = union_length_1d(X, c2)
            ly1 = union_length_1d(Y, c1)
            ly2 = union_length_1d(Y, c2)
            
            # Integrate product of two linear functions over [c1, c2]
            # f(u) = (1-u)lx1 + u lx2
            # g(u) = (1-u)ly1 + u ly2
            # u = (c - c1) / (c2 - c1)
            # Integral = (c2 - c1) * integral_0^1 f(u)g(u) du
            # f(u)g(u) = lx1*ly1 + u(lx2*ly1 + lx1*ly2 - 2*lx1*ly1) + u^2(lx1*ly1 - lx2*ly1 - lx1*ly2 + lx2*ly2)
            # Let A = lx1, B = lx2 - lx1, C = ly1, D = ly2 - ly1
            # f = A + Bu, g = C + Du
            # fg = AC + (AD+BC)u + BD u^2
            # Int = AC + (AD+BC)/2 + BD/3
            
            A = lx1
            B = lx2 - lx1
            C = ly1
            D = ly2 - ly1
            
            integral = A * C + (A * D + B * C) / 2.0 + B * D / 3.0
            total_vol += (c2 - c1) * integral
            
        return total_vol

    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    x0 = keich.copy()
    y0 = keich.copy()
    p0 = np.concatenate([x0, y0])
    
    # Objective wrapper for 16D vector
    def objective(p):
        X = p[:N]
        Y = p[N:]
        return coupled_volume(X, Y)
    
    # Multi-restart optimization
    rng_init = np.random.default_rng(42)
    initial_points = [p0.copy()]
    
    # Add perturbations
    for _ in range(10):
        perturbed = p0 + rng_init.normal(0, 0.05, size=2*N)
        initial_points.append(perturbed)
        
    # Add diverse starts
    initial_points.append(np.linspace(0, 1, 2*N))
    initial_points.append(rng_init.uniform(-0.5, 1.5, size=2*N))
    
    best_p = p0.copy()
    best_val = objective(p0)
    
    for p_init in initial_points:
        if time.time() - start_time > time_limit - margin:
            break
        
        try:
            res = opt.minimize(objective, p_init, method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_p = res.x.copy()
        except Exception:
            continue
            
    # Final polish with tighter tolerance if time permits
    if time.time() - start_time < time_limit - margin:
        try:
            res_polish = opt.minimize(objective, best_p, method='Nelder-Mead',
                                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            val = res_polish.fun
            if val < best_val:
                best_val = val
                best_p = res_polish.x.copy()
        except Exception:
            pass
            
    x_opt = best_p[:N]
    y_opt = best_p[N:]
    
    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Sanity check
    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
