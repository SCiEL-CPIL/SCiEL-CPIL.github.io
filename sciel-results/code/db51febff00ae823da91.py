# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def _union_length(intervals):
    """
    Compute the length of the union of a list of intervals (start, end).
    intervals: list of (start, end) tuples.
    """
    if not intervals:
        return 0.0
    # Sort by start
    intervals = sorted(intervals, key=lambda x: x[0])
    total_len = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total_len += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total_len += current_end - current_start
    return total_len

def _compute_1d_integral(params_1d):
    """
    Compute the 1D integrated union length: ∫_0^1 L(c) dc
    where L(c) is the union length of N intervals at height c.
    params_1d: array of N values (x_0..x_{N-1}).
    """
    N = 8
    x = params_1d
    
    # Find breakpoints where the ordering of interval endpoints changes
    break_pts = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = (i - j) / N
            if abs(denom) > 1e-12:
                # x_i + c*i/N = x_j + c*j/N
                c = (x[j] - x[i]) / denom
                if -1e-9 < c < 1 + 1e-9:
                    c_clamped = max(0.0, min(1.0, c))
                    break_pts.append(c_clamped)
                # x_i + c*i/N = x_j + c*j/N + 1/N
                c = (x[j] - x[i] + 1.0/N) / denom
                if -1e-9 < c < 1 + 1e-9:
                    c_clamped = max(0.0, min(1.0, c))
                    break_pts.append(c_clamped)
                # x_i + c*i/N + 1/N = x_j + c*j/N
                c = (x[j] - x[i] - 1.0/N) / denom
                if -1e-9 < c < 1 + 1e-9:
                    c_clamped = max(0.0, min(1.0, c))
                    break_pts.append(c_clamped)
                # x_i + c*i/N + 1/N = x_j + c*j/N + 1/N
                c = (x[j] - x[i]) / denom
                if -1e-9 < c < 1 + 1e-9:
                    c_clamped = max(0.0, min(1.0, c))
                    break_pts.append(c_clamped)
    
    # Combine and sort breakpoints
    all_breaks = sorted(set(break_pts))
    if all_breaks[0] != 0.0:
        all_breaks.insert(0, 0.0)
    if all_breaks[-1] != 1.0:
        all_breaks.append(1.0)
    
    # Integrate piecewise using Simpson's rule (exact for quadratics)
    V = 0.0
    for k in range(len(all_breaks) - 1):
        c0 = all_breaks[k]
        c1 = all_breaks[k+1]
        c_mid = (c0 + c1) / 2.0
        
        def get_length(c):
            ix = [(x[i] + c*i/N, x[i] + c*i/N + 1.0/N) for i in range(N)]
            return _union_length(ix)
        
        l0 = get_length(c0)
        lm = get_length(c_mid)
        l1 = get_length(c1)
        
        # Simpson's rule
        V += (c1 - c0) / 6.0 * (l0 + 4*lm + l1)
        
    return V

def _compute_objective(params):
    """
    Compute the volume V = ∫_0^1 Lx(c) * Ly(c) dc.
    params: array of 16 values, first 8 are x_0..x_7, next 8 are y_0..y_7.
    """
    N = 8
    x = params[:8]
    y = params[8:16]
    
    # Find breakpoints for x
    break_x = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = (i - j) / N
            if abs(denom) > 1e-12:
                c = (x[j] - x[i]) / denom
                if -1e-9 < c < 1 + 1e-9:
                    break_x.append(max(0.0, min(1.0, c)))
                c = (x[j] - x[i] + 1.0/N) / denom
                if -1e-9 < c < 1 + 1e-9:
                    break_x.append(max(0.0, min(1.0, c)))
                c = (x[j] - x[i] - 1.0/N) / denom
                if -1e-9 < c < 1 + 1e-9:
                    break_x.append(max(0.0, min(1.0, c)))
    
    # Find breakpoints for y
    break_y = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = (i - j) / N
            if abs(denom) > 1e-12:
                c = (y[j] - y[i]) / denom
                if -1e-9 < c < 1 + 1e-9:
                    break_y.append(max(0.0, min(1.0, c)))
                c = (y[j] - y[i] + 1.0/N) / denom
                if -1e-9 < c < 1 + 1e-9:
                    break_y.append(max(0.0, min(1.0, c)))
                c = (y[j] - y[i] - 1.0/N) / denom
                if -1e-9 < c < 1 + 1e-9:
                    break_y.append(max(0.0, min(1.0, c)))
    
    # Combine and sort breakpoints
    all_breaks = sorted(set(break_x + break_y))
    if all_breaks[0] != 0.0:
        all_breaks.insert(0, 0.0)
    if all_breaks[-1] != 1.0:
        all_breaks.append(1.0)
    
    # Integrate piecewise
    V = 0.0
    for k in range(len(all_breaks) - 1):
        c0 = all_breaks[k]
        c1 = all_breaks[k+1]
        c_mid = (c0 + c1) / 2.0
        
        def get_lengths(c):
            ix = [(x[i] + c*i/N, x[i] + c*i/N + 1.0/N) for i in range(N)]
            iy = [(y[j] + c*j/N, y[j] + c*j/N + 1.0/N) for j in range(N)]
            return _union_length(ix), _union_length(iy)
        
        lx0, ly0 = get_lengths(c0)
        lxm, lym = get_lengths(c_mid)
        lx1, ly1 = get_lengths(c1)
        
        f0 = lx0 * ly0
        fm = lxm * lym
        f1 = lx1 * ly1
        
        # Simpson's rule (exact for quadratics)
        V += (c1 - c0) / 6.0 * (f0 + 4*fm + f1)
        
    return V

def _keich_offsets(N=8):
    """Generate Keich construction offsets."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = 600.0 - 10.0  # Safety margin
    
    # Initial guess: Keich construction
    keich = _keich_offsets(N)
    
    # Strategy: First optimize x and y independently to minimize their 1D integrals,
    # then use those as initial guess for joint optimization.
    
    # Step 1: Optimize x independently
    best_x = keich.copy()
    best_x_obj = _compute_1d_integral(best_x)
    
    x_options = [
        keich.copy(),
        -keich.copy(),
        np.sort(keich),
        np.sort(-keich),
        keich + np.linspace(0, 0.1, N),
        keich - np.linspace(0, 0.1, N),
    ]
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        x_options.append(keich + rng.uniform(-0.15, 0.15, N))
    
    for x_init in x_options:
        if time.time() > deadline - 60:
            break
        try:
            res = scipy.optimize.minimize(_compute_1d_integral, x_init, method='L-BFGS-B',
                                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res.fun < best_x_obj:
                best_x_obj = res.fun
                best_x = res.x
        except Exception:
            continue
    
    # Step 2: Optimize y independently (same as x due to symmetry, but use best_x as start)
    best_y = best_x.copy()
    best_y_obj = _compute_1d_integral(best_y)
    
    y_options = [
        best_x.copy(),
        -best_x.copy(),
        np.sort(best_x),
        np.sort(-best_x),
    ]
    for _ in range(5):
        y_options.append(best_x + rng.uniform(-0.15, 0.15, N))
    
    for y_init in y_options:
        if time.time() > deadline - 40:
            break
        try:
            res = scipy.optimize.minimize(_compute_1d_integral, y_init, method='L-BFGS-B',
                                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res.fun < best_y_obj:
                best_y_obj = res.fun
                best_y = res.x
        except Exception:
            continue
    
    # Step 3: Joint optimization starting from (best_x, best_y)
    params0 = np.concatenate([best_x, best_y])
    best_params = params0.copy()
    best_obj = _compute_objective(params0)
    
    # Joint optimization with multiple restarts
    joint_options = [
        params0.copy(),
        np.concatenate([best_x, -best_y]),
        np.concatenate([-best_x, best_y]),
        np.concatenate([best_y, best_x]),
        np.concatenate([best_x, best_x]),
        np.concatenate([best_y, best_y]),
    ]
    
    # Add perturbations
    for _ in range(8):
        pert = rng.uniform(-0.1, 0.1, 2*N)
        joint_options.append(params0 + pert)
    
    # Add diverse starts
    joint_options.append(np.concatenate([keich, keich]))
    joint_options.append(np.concatenate([keich, -keich]))
    joint_options.append(np.concatenate([rng.uniform(0, 1, N), rng.uniform(0, 1, N)]))
    
    for p_init in joint_options:
        if time.time() > deadline - 20:
            break
        try:
            res = scipy.optimize.minimize(_compute_objective, p_init, method='L-BFGS-B',
                                         options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            continue
    
    # Step 4: Final polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            res_polish = scipy.optimize.minimize(_compute_objective, best_params, method='L-BFGS-B',
                                                 options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False})
            if res_polish.fun < best_obj:
                best_obj = res_polish.fun
                best_params = res_polish.x
        except Exception:
            pass
    
    x_opt = best_params[:8]
    y_opt = best_params[8:16]
    
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
