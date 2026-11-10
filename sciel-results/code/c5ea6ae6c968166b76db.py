# EVOLVE-BLOCK-START
import numpy as np
import time


def _mc_volume(positions, rng_seed=12345, n_samples=500000):
    """Compute MC estimate of union volume for given 64x2 base positions."""
    N = 8
    pos = positions.reshape(N, N, 2)
    # pos[i,j,0] = x_base, pos[i,j,1] = y_base
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    
    # Compute box bounds
    # X0 = min_k x_k, X1 = max_k (x_k + (i+1)/8)
    # Y0 = min_k y_k, Y1 = max_k (y_k + (j+1)/8)
    x_bases = pos[:, :, 0]  # shape (N, N)
    y_bases = pos[:, :, 1]
    
    # For each tube (i,j): x range at top is [x_base + i/8, x_base + i/8 + 1/8]
    # The box X0 = min over all k of x_k (base x)
    # X1 = max over all k of (x_k + (i+1)/8) where i = k//8
    # So for tube (i,j), x_k = x_bases[i,j], and x_k + (i+1)/8 = x_bases[i,j] + (i+1)/8
    
    X0 = np.min(x_bases)
    Y0 = np.min(y_bases)
    
    # X1: for each (i,j), x_bases[i,j] + (i+1)/8
    i_offsets = (i_idx + 1) / N  # shape (N,)
    x_top = x_bases + i_offsets[:, np.newaxis]  # shape (N, N)
    X1 = np.max(x_top)
    
    j_offsets = (j_idx + 1) / N  # shape (N,)
    y_top = y_bases + j_offsets[np.newaxis, :]  # shape (N, N)
    Y1 = np.max(y_top)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 1e10  # invalid, return large value
    
    rng = np.random.default_rng(rng_seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if point (a, b, c) is inside at least one tube
    # Tube (i,j): [x_base[i,j] + c*i/N, x_base[i,j] + c*i/N + 1/N] x [y_base[i,j] + c*j/N, y_base[i,j] + c*j/N + 1/N]
    # Vectorize: for each tube, check all points
    # a shape (n_samples,), c shape (n_samples,)
    # For tube (i,j): x_lo = x_bases[i,j] + c * i/N, x_hi = x_lo + 1/N
    #                  y_lo = y_bases[i,j] + c * j/N, y_hi = y_lo + 1/N
    
    # Build arrays for all tubes
    # x_bases shape (N, N) -> flatten to (64,)
    xb_flat = x_bases.flatten()  # (64,)
    yb_flat = y_bases.flatten()  # (64,)
    i_flat = np.repeat(i_idx, N)  # (64,)
    j_flat = np.tile(j_idx, N)  # (64,)
    
    # For each tube k (0..63) and each sample m (0..n_samples-1):
    # x_lo[k,m] = xb_flat[k] + c[m] * i_flat[k] / N
    # x_hi[k,m] = x_lo[k,m] + 1/N
    # y_lo[k,m] = yb_flat[k] + c[m] * j_flat[k] / N
    # y_hi[k,m] = y_lo[k,m] + 1/N
    
    # To avoid 64 x 500000 array (32M floats = 256MB), process in chunks
    chunk_size = 50000
    inside_any = np.zeros(n_samples, dtype=bool)
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        a_chunk = a[start:end]  # (chunk,)
        b_chunk = b[start:end]  # (chunk,)
        c_chunk = c[start:end]  # (chunk,)
        
        # x_lo: (64, chunk), x_hi: (64, chunk)
        x_lo = xb_flat[:, np.newaxis] + c_chunk[np.newaxis, :] * i_flat[:, np.newaxis] / N
        x_hi = x_lo + 1.0 / N
        y_lo = yb_flat[:, np.newaxis] + c_chunk[np.newaxis, :] * j_flat[:, np.newaxis] / N
        y_hi = y_lo + 1.0 / N
        
        in_x = (a_chunk[np.newaxis, :] >= x_lo) & (a_chunk[np.newaxis, :] <= x_hi)  # (64, chunk)
        in_y = (b_chunk[np.newaxis, :] >= y_lo) & (b_chunk[np.newaxis, :] <= y_hi)  # (64, chunk)
        
        in_tube = in_x & in_y  # (64, chunk)
        inside_any[start:end] = np.any(in_tube, axis=0)
    
    fraction = np.mean(inside_any)
    V = fraction * box_area
    return V


def _objective(params, n_samples=500000):
    """Objective function for Nelder-Mead: returns MC volume estimate."""
    positions = params.reshape(64, 2)
    return _mc_volume(positions, rng_seed=12345, n_samples=n_samples)


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with N=8.
    Uses Nelder-Mead optimization with multiple restarts to minimize
    the Monte Carlo estimate of the union volume of 64 sheared tubes.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 10 minute limit, leave margin
    
    N = 8
    n_tubes = N * N  # 64
    
    # Initial guess: use Keich-like construction as starting point
    pos_log = int(np.log2(N))
    j_terms = np.arange(1, pos_log + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_log - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_log)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    init_positions = np.stack([base_x, base_y], axis=1)
    
    best_positions = init_positions.copy()
    best_volume = _mc_volume(best_positions, rng_seed=12345, n_samples=500000)
    
    # Nelder-Mead with multiple restarts
    from scipy.optimize import minimize
    
    n_samples_opt = 100000  # fewer samples for speed during optimization
    n_restarts = 20  # number of random restarts
    
    # Precompute time budget
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 30.0  # 30s safety margin
    
    if remaining <= 0:
        return best_positions
    
    # Try different n_samples based on available time
    # Estimate time per evaluation
    t0 = time.time()
    _ = _objective(best_positions.flatten(), n_samples=n_samples_opt)
    t_eval = time.time() - t0
    
    # Nelder-Mead typically needs ~100-200 function evaluations per restart
    # With 64 parameters, each eval takes ~t_eval seconds
    # We can afford: remaining / (t_eval * 150) restarts
    max_restarts = max(1, int(remaining / (t_eval * 150)))
    n_restarts = min(n_restarts, max_restarts)
    
    # Also reduce n_samples if we have many restarts
    if n_restarts > 5:
        n_samples_opt = min(n_samples_opt, 50000)
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 60.0:  # 60s safety margin
            break
        
        # Create perturbed initial position
        if restart == 0:
            x0 = best_positions.flatten().copy()
        else:
            # Random perturbation around best solution
            scale = 0.05
            x0 = best_positions.flatten() + rng.normal(0, scale, size=128)
        
        # Nelder-Mead options
        options = {
            'maxiter': 300,
            'xatol': 1e-6,
            'fatol': 1e-8,
            'adaptive': True,
        }
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options=options,
                args=(n_samples_opt,)
            )
            
            if result.fun < best_volume:
                best_volume = result.fun
                best_positions = result.x.reshape(64, 2)
        except Exception:
            continue
    
    # Final evaluation with full sample count
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        final_volume = _mc_volume(best_positions, rng_seed=12345, n_samples=500000)
        if final_volume < best_volume:
            best_volume = final_volume
    
    # Validate result
    if not np.all(np.isfinite(best_positions)):
        # Fallback to initial
        return init_positions
    
    return best_positions


# EVOLVE-BLOCK-END
