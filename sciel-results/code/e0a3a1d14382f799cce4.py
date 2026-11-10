import numpy as np
import time

def _compute_volume(positions, n_c=64, n_xy=256):
    """
    Fast 3D voxel-based volume estimate matching the evaluator's Monte Carlo method.
    
    Positions: (64, 2) array of (x_k, y_k).
    Tube (i,j) at height c occupies:
      x in [x_k + c*i/8, x_k + c*i/8 + 1/8]
      y in [y_k + c*j/8, y_k + c*j/8 + 1/8]
    """
    N = 8
    delta = 1.0 / N
    K = 64
    
    # Get box bounds
    i_idx = np.arange(K) // N
    j_idx = np.arange(K) % N
    x_pos = positions[:, 0]
    y_pos = positions[:, 1]
    
    # X range: x_k to x_k + (i+1)/8
    x_min_all = x_pos.copy()
    x_max_all = x_pos + (i_idx + 1) * delta
    X0 = x_min_all.min()
    X1 = x_max_all.max()
    
    # Y range: y_k to y_k + (j+1)/8
    y_min_all = y_pos.copy()
    y_max_all = y_pos + (j_idx + 1) * delta
    Y0 = y_min_all.min()
    Y1 = y_max_all.max()
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return np.inf
    
    # Voxel grid
    dx = (X1 - X0) / n_xy
    dy = (Y1 - Y0) / n_xy
    dc = 1.0 / n_c
    
    # Voxel centers
    xs = X0 + (np.arange(n_xy) + 0.5) * dx
    ys = Y0 + (np.arange(n_xy) + 0.5) * dy
    cs = (np.arange(n_c) + 0.5) * dc
    
    # For each voxel, check if inside any tube
    # Vectorize over x,y; loop over c for memory efficiency
    total_volume = 0.0
    voxel_area = dx * dy
    
    # Precompute tube parameters
    # Tube k at height c: x in [x_k + c*i_k/8, x_k + c*i_k/8 + 1/8]
    #                     y in [y_k + c*j_k/8, y_k + c*j_k/8 + 1/8]
    
    for ci in range(n_c):
        c = cs[ci]
        # For each tube, compute x-range and y-range at this c
        x_lo = x_pos + c * i_idx * delta  # (K,)
        x_hi = x_lo + delta
        y_lo = y_pos + c * j_idx * delta  # (K,)
        y_hi = y_lo + delta
        
        # Check each voxel
        # (n_xy, n_xy) boolean grid
        # For efficiency, use broadcasting: xs[None,:] vs x_lo[:,None]
        # But K=64, n_xy=256, so 64*256*256 = 4M ops per c, times 64 c = 256M total
        # That's a lot. Let's use a smarter approach.
        
        # Alternative: for each voxel, check all tubes
        # Actually let's just do it with numpy broadcasting in chunks
        
        # Create meshgrid for x,y
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        XX_flat = XX.ravel()  # (n_xy*n_xy,)
        YY_flat = YY.ravel()
        
        # For each tube, check if voxel is inside
        # inside[k, m] = (XX_flat[m] >= x_lo[k]) & (XX_flat[m] <= x_hi[k]) & 
        #                 (YY_flat[m] >= y_lo[k]) & (YY_flat[m] <= y_hi[k])
        # This is K x (n_xy*n_xy) = 64 x 65536 = 4M, manageable
        
        # Vectorized: 
        # x_inside: (K, M) where M = n_xy*n_xy
        x_in = (XX_flat[None, :] >= x_lo[:, None]) & (XX_flat[None, :] <= x_hi[:, None])
        y_in = (YY_flat[None, :] >= y_lo[:, None]) & (YY_flat[None, :] <= y_hi[:, None])
        inside = x_in & y_in  # (K, M)
        
        # A voxel is covered if any tube covers it
        covered = inside.any(axis=0)  # (M,)
        total_volume += covered.sum() * voxel_area * dc
    
    return total_volume


def kakeya_needle_3d8() -> np.ndarray:
    """
    Returns a (64, 2) array of base positions for a 3D Kakeya set made of 64 sheared tubes.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s to leave safety margin
    
    N = 8
    delta = 1.0 / N
    K = 64
    
    # Initial positions: centered heuristic
    positions = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):
        i = k // N
        j = k % N
        positions[k, 0] = -i / (2.0 * N)
        positions[k, 1] = -j / (2.0 * N)
    
    best_positions = positions.copy()
    best_vol = _compute_volume(best_positions)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_positions
    
    # Nelder-Mead optimization over all 64*2 = 128 variables
    # Use scipy if available, otherwise implement simple version
    try:
        from scipy.optimize import minimize
        import scipy.optimize
        
        # Flatten positions
        x0 = best_positions.ravel()
        
        def objective(flat_pos):
            pos = flat_pos.reshape(K, 2)
            return _compute_volume(pos)
        
        # Nelder-Mead with relaxed tolerances
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 200,
                'maxfev': 500,
                'xatol': 1e-4,
                'fatol': 1e-6,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < best_vol:
            new_pos = result.x.reshape(K, 2)
            new_vol = result.fun
            if np.all(np.isfinite(new_pos)) and new_vol > 0:
                best_positions = new_pos
                best_vol = new_vol
        
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_positions
    
    # Try a second round with tighter tolerance if time permits
    try:
        from scipy.optimize import minimize
        
        x0 = best_positions.ravel()
        
        def objective(flat_pos):
            pos = flat_pos.reshape(K, 2)
            return _compute_volume(pos)
        
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 150,
                'maxfev': 400,
                'xatol': 1e-5,
                'fatol': 1e-7,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < best_vol:
            new_pos = result.x.reshape(K, 2)
            new_vol = result.fun
            if np.all(np.isfinite(new_pos)) and new_vol > 0:
                best_positions = new_pos
                best_vol = new_vol
                
    except Exception:
        pass
    
    # Final validation
    if not np.all(np.isfinite(best_positions)):
        # Fallback to initial
        for k in range(K):
            i = k // N
            j = k % N
            best_positions[k, 0] = -i / (2.0 * N)
            best_positions[k, 1] = -j / (2.0 * N)
    
    return best_positions
