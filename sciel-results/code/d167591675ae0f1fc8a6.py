# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

DEADLINE = 600.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _time_left():
    return DEADLINE - _elapsed() - 30.0  # 30s safety margin


def _integrate(y, x):
    """
    Compatible trapezoidal integration.
    """
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    else:
        return np.trapz(y, x)


def _compute_1d_union_length(positions, N):
    """
    Compute the total length of the union of intervals in 1D.
    """
    n_pts = 200
    c_vals = np.linspace(0, 1, n_pts)
    lengths = np.zeros(n_pts)
    for idx, c in enumerate(c_vals):
        starts = positions + c * np.arange(N) / N
        ends = starts + 1.0 / N
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        total = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] > cur_end:
                total += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
            else:
                if e_sorted[k] > cur_end:
                    cur_end = e_sorted[k]
        total += cur_end - cur_start
        lengths[idx] = total
    
    return _integrate(lengths, c_vals)


def _compute_1d_union_length_fast(positions, N, n_c=50):
    """Faster version with fewer integration points."""
    c_vals = np.linspace(0, 1, n_c)
    lengths = np.zeros(n_c)
    arange_N = np.arange(N) / N
    for idx in range(n_c):
        c = c_vals[idx]
        starts = positions + c * arange_N
        ends = starts + 1.0 / N
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        total = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] > cur_end:
                total += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
            else:
                if e_sorted[k] > cur_end:
                    cur_end = e_sorted[k]
        total += cur_end - cur_start
        lengths[idx] = total
    return _integrate(lengths, c_vals)


def _keich_offsets(N):
    """Generate Keich binary construction offsets for N positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _evaluate_positions(positions, N):
    """
    Evaluate the 3D union volume for given base positions.
    positions: shape (N*N, 2), row k = tube (i,j) with i=k//N, j=k%N.
    Returns the estimated volume V.
    """
    i_vals = (np.arange(N * N) // N).astype(np.float64)
    j_vals = (np.arange(N * N) % N).astype(np.float64)
    
    x_base = positions[:, 0]
    y_base = positions[:, 1]
    
    # X0 = min x_k, X1 = max (x_k + (i+1)/8)
    # At height c, x-interval is [x_k + c*i/N, x_k + c*i/N + 1/N]
    # The full x-range for tube k is [x_k, x_k + 1/N + i/N]
    # So X1 = max_k (x_k + 1/N + i_k/N), X0 = min_k x_k
    x_extents = x_base + 1.0 / N + i_vals / N
    X0 = np.min(x_base)
    X1 = np.max(x_extents)
    
    y_extents = y_base + 1.0 / N + j_vals / N
    Y0 = np.min(y_base)
    Y1 = np.max(y_extents)
    
    if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
        return np.inf
    
    # Monte Carlo estimate
    rng = np.random.default_rng(12345)
    n_mc = 500000
    a = rng.uniform(X0, X1, n_mc)
    b = rng.uniform(Y0, Y1, n_mc)
    c = rng.uniform(0, 1, n_mc)
    
    # Check if each point is inside at least one tube
    # Tube k at height c: x in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N], y in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    # Vectorized check: for each point, check against all 64 tubes
    # This is 500000 x 64 which might be memory heavy. Use chunked approach.
    
    chunk_size = 50000
    inside_any = np.zeros(n_mc, dtype=bool)
    for start in range(0, n_mc, chunk_size):
        end = min(start + chunk_size, n_mc)
        a_chunk = a[start:end]
        b_chunk = b[start:end]
        c_chunk = c[start:end]
        
        x_low = x_base + c_chunk[:, None] * i_vals[None, :] / N
        x_high = x_low + 1.0 / N
        y_low = y_base + c_chunk[:, None] * j_vals[None, :] / N
        y_high = y_low + 1.0 / N
        
        in_x = (a_chunk[:, None] >= x_low) & (a_chunk[:, None] <= x_high)
        in_y = (b_chunk[:, None] >= y_low) & (b_chunk[:, None] <= y_high)
        inside_any[start:end] = np.any(in_x & in_y, axis=1)
    
    fraction = np.mean(inside_any)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    return V


def _optimize_1d_keich_init(N, max_iters=200):
    """
    Optimize 1D Keich-type positions using Nelder-Mead.
    Returns optimized positions array of length N.
    """
    keich = _keich_offsets(N)
    x0 = keich.copy()
    
    def cost(x):
        return _compute_1d_union_length_fast(x, N, n_c=50)
    
    result = minimize(cost, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x


def _optimize_2d_joint(N, x_init, y_init, max_iters=100, budget=120.0):
    """
    Joint optimization of both x and y positions using Nelder-Mead.
    The objective is the full 3D volume (approximated via product of 1D lengths times box area correction).
    
    Actually, the 3D volume is NOT simply the product of 1D union lengths because the union
    in 3D is not a product set in general. However, if x and y are independent (x depends only on i,
    y depends only on j), then the union IS a product of the 1D unions, and V = L_x * L_y where
    L_x is the 1D union length for x-direction and L_y for y-direction.
    
    So we can optimize L_x and L_y independently!
    """
    # Optimize x positions
    result_x = minimize(lambda x: _compute_1d_union_length_fast(x, N, n_c=50), 
                        x_init, method='Nelder-Mead',
                        options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12})
    
    # Optimize y positions
    result_y = minimize(lambda y: _compute_1d_union_length_fast(y, N, n_c=50), 
                        y_init, method='Nelder-Mead',
                        options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12})
    
    return result_x.x, result_y.x


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns positions of shape (64, 2).
    """
    N = 8
    keich = _keich_offsets(N)
    
    # Optimize 1D positions using Nelder-Mead with a reasonable budget
    # The Keich construction is a strong initial guess.
    # We perform a limited number of restarts to ensure we finish in time.
    
    best_x = keich.copy()
    best_cost_x = _compute_1d_union_length_fast(keich, N, n_c=100)
    
    rng_restart = np.random.default_rng(42)
    
    # Use time budget to determine how many restarts we can afford
    # Each Nelder-Mead run with 100 integration points takes ~0.1-0.5s
    # We reserve 60s for final evaluation and safety.
    
    max_restarts = 5
    max_nm_iters = 200
    
    for r in range(max_restarts):
        if _time_left() < 90:
            break
        
        if r == 0:
            x0 = keich.copy()
        else:
            x0 = keich + rng_restart.normal(0, 0.1, size=N)
        
        try:
            result = minimize(lambda x: _compute_1d_union_length_fast(x, N, n_c=100),
                             x0, method='Nelder-Mead',
                             options={'maxiter': max_nm_iters, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_cost_x:
                best_cost_x = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Use same optimal 1D structure for y (by symmetry)
    best_y = best_x.copy()
    
    # Build full 64x2 position array
    i_vals = np.repeat(np.arange(N), N)
    j_vals = np.tile(np.arange(N), N)
    
    base_x = best_x[i_vals]
    base_y = best_y[j_vals]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Verify validity
    V = _evaluate_positions(positions, N)
    if not np.isfinite(V) or V <= 0:
        # Fallback to Keich
        base_x = keich[i_vals]
        base_y = keich[j_vals]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
