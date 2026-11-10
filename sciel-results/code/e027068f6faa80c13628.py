import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import logsumexp

def _merge_intervals(Ls, Rs):
    """Merge intervals and return (merged_list, total_length, left_indices, right_indices)."""
    n = len(Ls)
    # Create list of (left, right, left_idx, right_idx)
    intervals = [(Ls[i], Rs[i], i, i) for i in range(n)]
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    cur_l, cur_r, cur_li, cur_ri = intervals[0]
    for l, r, li, ri in intervals[1:]:
        if l > cur_r + 1e-15:
            merged.append((cur_l, cur_r, cur_li, cur_ri))
            cur_l, cur_r, cur_li, cur_ri = l, r, li, ri
        else:
            if r > cur_r:
                cur_r = r
                cur_ri = ri
    merged.append((cur_l, cur_r, cur_li, cur_ri))
    
    total_len = sum(r - l for l, r, _, _ in merged)
    return merged, total_len

def _get_breakpoints(X, N=8):
    """Calculate all c in (0, 1) where the order of endpoints changes."""
    breakpoints = set([0.0, 1.0])
    for i in range(N):
        for j in range(i + 1, N):
            denom = float(i - j)
            # L_i = L_j: X[i] + i*c/N = X[j] + j*c/N => c = N*(X[j]-X[i])/(i-j)
            c1 = N * (X[j] - X[i]) / denom
            if 0.0 < c1 < 1.0:
                breakpoints.add(c1)
            # L_i = R_j: X[i] + i*c/N = X[j] + j*c/N + 1/N
            c2 = N * (X[j] + 1.0/N - X[i]) / denom
            if 0.0 < c2 < 1.0:
                breakpoints.add(c2)
            # R_i = L_j: X[i] + i*c/N + 1/N = X[j] + j*c/N
            c3 = N * (X[j] - X[i] - 1.0/N) / denom
            if 0.0 < c3 < 1.0:
                breakpoints.add(c3)
            # R_i = R_j: X[i] + i*c/N + 1/N = X[j] + j*c/N + 1/N => same as L_i=L_j
            c4 = N * (X[j] - X[i]) / denom
            if 0.0 < c4 < 1.0:
                breakpoints.add(c4)
    return sorted(breakpoints)

def _compute_1d_cost_squared(X, N=8):
    """
    Compute integral of (Union Length)^2 over c in [0, 1].
    This equals the 3D union volume for the sheared tube construction.
    """
    if not np.all(np.isfinite(X)):
        return 1e10
    
    breakpoints = _get_breakpoints(X, N)
    
    total_integral = 0.0
    
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        if c_end - c_start < 1e-15:
            continue
            
        c_mid = 0.5 * (c_start + c_end)
        
        # Endpoints at c_mid
        Ls = np.array([X[i] + (i / N) * c_mid for i in range(N)])
        Rs = np.array([X[i] + (i / N) * c_mid + 1.0 / N for i in range(N)])
        
        merged, L_mid = _merge_intervals(Ls, Rs)
        
        # Calculate exact slope of union length
        slope = 0.0
        for l, r, li, ri in merged:
            slope += (ri - li) / N
        
        h = (c_end - c_start) / 2.0
        A = slope
        B = L_mid
        integral_segment = (2.0 * A * A * h**3) / 3.0 + 2.0 * B * B * h
        total_integral += integral_segment
    
    return total_integral

def _compute_1d_cost_smooth(X, N=8, alpha=50.0):
    """
    Smooth approximation of the 1D cost using soft-min for union length.
    This provides a differentiable surrogate for gradient-based optimization.
    """
    if not np.all(np.isfinite(X)):
        return 1e10
    
    # Sample c values densely
    n_samples = 200
    c_vals = np.linspace(0.01, 0.99, n_samples)
    
    # Compute Ls and Rs for all tubes at all c values
    Ls = X[None, :] + (np.arange(N)[None, :] / N) * c_vals[:, None]  # shape (n_samples, N)
    Rs = Ls + 1.0 / N
    
    # Union length at each c: soft-min of right endpoints minus soft-min of left endpoints
    # Actually, union length = max(Rs) - min(Ls) only if all intervals overlap
    # More generally, we need to compute the actual union length
    
    # For smooth approximation, use the fact that union length is piecewise linear
    # and approximate it with a smooth function
    
    # Compute union length exactly at each sample point, then smooth
    union_lengths = np.zeros(n_samples)
    for idx, c_val in enumerate(c_vals):
        Ls_c = Ls[idx]
        Rs_c = Rs[idx]
        _, L_union = _merge_intervals(Ls_c, Rs_c)
        union_lengths[idx] = L_union
    
    # Integrate (union_length)^2 using trapezoidal rule
    integral = np.trapz(union_lengths ** 2, c_vals)
    
    return integral

def _compute_1d_cost_gradient(X, N=8, eps=1e-7):
    """
    Compute gradient of the 1D cost function via finite differences.
    """
    if not np.all(np.isfinite(X)):
        return None
    
    grad = np.zeros(N)
    f0 = _compute_1d_cost_squared(X, N)
    
    for i in range(N):
        X_plus = X.copy()
        X_plus[i] += eps
        f_plus = _compute_1d_cost_squared(X_plus, N)
        
        X_minus = X.copy()
        X_minus[i] -= eps
        f_minus = _compute_1d_cost_squared(X_minus, N)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad

def _compute_volume_mc(positions, N=8, n_samples=50000):
    """
    Compute union volume using Monte Carlo estimation for verification.
    """
    if positions.shape != (N * N, 2):
        return 1e10
    
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    
    i_vals = np.repeat(np.arange(N), N)
    j_vals = np.tile(np.arange(N), N)
    
    X0 = np.min(x_coords)
    X1 = np.max(x_coords + (i_vals + 1) / N)
    Y0 = np.min(y_coords)
    Y1 = np.max(y_coords + (j_vals + 1) / N)
    
    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return 1e10
    
    rng = np.random.default_rng(42)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    inside = np.zeros(n_samples, dtype=bool)
    
    for k in range(N * N):
        i_k = k // N
        j_k = k % N
        x_k = x_coords[k]
        y_k = y_coords[k]
        
        a_min = x_k + c * i_k / N
        a_max = a_min + 1.0 / N
        b_min = y_k + c * j_k / N
        b_max = b_min + 1.0 / N
        
        mask = (a >= a_min) & (a <= a_max) & (b >= b_min) & (b <= b_max)
        inside |= mask
    
    fraction = np.mean(inside)
    volume = fraction * area
    return volume

def _build_positions(X, N=8):
    """Build the 64x2 positions array from the 1D coordinate array X."""
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X[i_flat]
    base_y = X[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570  # Safety margin: 60s before 630s kill
    
    N = 8
    
    # Phase 1: Generate diverse initializations
    inits = [
        np.zeros(N),
        np.arange(N) / N,
        -np.arange(N) / N,
        np.random.default_rng(12345).uniform(-0.5, 0.5, N),
        np.array([0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]) / N,
        np.array([0, 0.25, 0.5, 0.75, 0, 0.25, 0.5, 0.75]) / N,
        np.array([0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5]) / N,
        np.linspace(0, 0.5, N),
        -np.linspace(0, 0.5, N),
        np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]) / N,
        np.array([0, 1, 0, 1, 0, 1, 0, 1]) / N,
        np.array([0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5]) / N,
    ]
    
    best_X = None
    best_cost = float('inf')
    
    # Phase 1: Nelder-Mead from multiple starts
    for init in inits:
        if time.time() > deadline - 180:
            break
            
        try:
            res = minimize(_compute_1d_cost_squared, init, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass
    
    if best_X is None:
        best_X = np.zeros(N)
    
    # Phase 2: L-BFGS-B with gradient (more efficient for smooth optimization)
    if time.time() < deadline - 120:
        try:
            def obj_with_grad(X):
                f = _compute_1d_cost_squared(X, N)
                g = _compute_1d_cost_gradient(X, N)
                if g is None:
                    return f, None
                return f, g
            
            res = minimize(obj_with_grad, best_X, method='L-BFGS-B', 
                           jac=True,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Refine with tighter Nelder-Mead
    if time.time() < deadline - 60:
        try:
            res = minimize(_compute_1d_cost_squared, best_X, method='Nelder-Mead', 
                           options={'maxiter': 15000, 'xatol': 1e-13, 'fatol': 1e-13})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Powell refinement
    if time.time() < deadline - 30:
        try:
            res = minimize(_compute_1d_cost_squared, best_X, method='Powell', 
                           options={'maxiter': 8000, 'xtol': 1e-12, 'ftol': 1e-12})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res = minimize(_compute_1d_cost_squared, best_X, method='Nelder-Mead', 
                           options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-15})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass
    
    # Construct final positions
    positions = _build_positions(best_X, N)
    
    # Verify finiteness
    if not np.all(np.isfinite(positions)):
        positions = _build_positions(np.zeros(N), N)
    
    return positions
