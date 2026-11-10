import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _compute_volume_fast(positions, n_c=50, n_xy=80):
    """
    Fast approximate volume computation using deterministic sampling.
    positions: (64, 2) array of base positions.
    Returns estimated volume of union.
    """
    N = 8
    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)
    
    x_base = positions[:, 0]
    y_base = positions[:, 1]
    
    # Compute bounding box
    x_max = x_base + (i_flat + 1) / N
    y_max = y_base + (j_flat + 1) / N
    X0 = np.min(x_base)
    X1 = np.max(x_max)
    Y0 = np.min(y_base)
    Y1 = np.max(y_max)
    
    area_box = (X1 - X0) * (Y1 - Y0)
    if area_box <= 1e-9:
        return 1e10
    
    # Sample points
    c_vals = np.linspace(0.01, 0.99, n_c)
    a_vals = np.linspace(X0 + 1e-6, X1 - 1e-6, n_xy)
    b_vals = np.linspace(Y0 + 1e-6, Y1 - 1e-6, n_xy)
    
    # Use broadcasting: for each c, check if each (a,b) is in any tube
    # Tube k at height c: x in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
    #                    y in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    
    total_hits = 0.0
    total_samples = 0.0
    
    for c in c_vals:
        # For each tube, compute the x and y intervals at this height
        x_low = x_base + c * i_flat / N
        x_high = x_low + 1.0 / N
        y_low = y_base + c * j_flat / N
        y_high = y_low + 1.0 / N
        
        # Check if any tube contains the point
        # a_vals shape (n_xy,), b_vals shape (n_xy,)
        # x_low shape (64,)
        # We need: for each (a_i, b_j), is there a tube k such that
        #   a_i in [x_low_k, x_high_k] AND b_j in [y_low_k, y_high_k]
        
        # Vectorized: create 2D array of a and b
        A = a_vals[:, None]  # (n_xy, 1)
        B = b_vals[None, :]  # (1, n_xy)
        
        # For each tube, check membership
        # x_in: (n_xy, 64) - is a_i in [x_low_k, x_high_k]
        x_in = (A >= x_low[None, :]) & (A <= x_high[None, :])  # (n_xy, 64)
        y_in = (B >= y_low[None, :]) & (B <= y_high[None, :])  # (1, n_xy, 64)
        
        # Combine: point (i,j) is hit if exists k with x_in[i,k] and y_in[j,k]
        # y_in is (1, n_xy, 64), x_in is (n_xy, 64)
        # Need: for each (i, j), exists k: x_in[i, k] AND y_in[0, j, k]
        # Create (n_xy, n_xy, 64) - too large? n_xy=80, 80*80*64 = 409600, ok
        hit = np.any(x_in[:, None, :] & y_in, axis=2)  # (n_xy, n_xy)
        
        total_hits += np.sum(hit)
        total_samples += n_xy * n_xy
    
    fraction = total_hits / total_samples
    return fraction * area_box


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    n_tubes = N * N
    deadline = time.time() + 600.0
    
    # Initial position: Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x_init = keich[i_flat].copy()
    base_y_init = keich[j_flat].copy()
    positions_init = np.stack([base_x_init, base_y_init], axis=1)
    
    best_positions = positions_init.copy()
    best_volume = _compute_volume_fast(best_positions)
    
    # Try several optimization strategies
    strategies = []
    
    # Strategy 1: L-BFGS-B from Keich init
    strategies.append(('lBFGS_B', positions_init.copy()))
    
    # Strategy 2: Powell from Keich init
    strategies.append(('Powell', positions_init.copy()))
    
    # Strategy 3: Nelder-Mead from Keich init
    strategies.append(('Nelder-Mead', positions_init.copy()))
    
    # Strategy 4: SLSQP from Keich init
    strategies.append(('SLSQP', positions_init.copy()))
    
    for method_name, start_pos in strategies:
        if time.time() > deadline - 60:
            break
        
        def objective(pos_flat):
            if time.time() > deadline - 30:
                return 1e10
            pos = pos_flat.reshape(n_tubes, 2)
            if not np.all(np.isfinite(pos)):
                return 1e10
            vol = _compute_volume_fast(pos)
            if not np.isfinite(vol) or vol <= 0:
                return 1e10
            return vol
        
        try:
            res = minimize(
                objective,
                start_pos.flatten(),
                method=method_name,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_volume:
                new_pos = res.x.reshape(n_tubes, 2)
                if np.all(np.isfinite(new_pos)):
                    new_vol = _compute_volume_fast(new_pos)
                    if np.isfinite(new_vol) and new_vol > 0 and new_vol < best_volume:
                        best_volume = new_vol
                        best_positions = new_pos
        except Exception:
            pass
    
    # Strategy 5: Multi-start with small perturbations
    if time.time() < deadline - 120:
        rng = np.random.default_rng(42)
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline - 60:
                break
            
            # Perturb the best position
            perturbation_scale = 0.05 / (r + 1)
            perturbed = best_positions + rng.normal(0, perturbation_scale, size=(n_tubes, 2))
            
            def objective2(pos_flat):
                if time.time() > deadline - 30:
                    return 1e10
                pos = pos_flat.reshape(n_tubes, 2)
                if not np.all(np.isfinite(pos)):
                    return 1e10
                vol = _compute_volume_fast(pos)
                if not np.isfinite(vol) or vol <= 0:
                    return 1e10
                return vol
            
            try:
                res = minimize(
                    objective2,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'ftol': 1e-10}
                )
                
                if res.fun < best_volume:
                    new_pos = res.x.reshape(n_tubes, 2)
                    if np.all(np.isfinite(new_pos)):
                        new_vol = _compute_volume_fast(new_pos)
                        if np.isfinite(new_vol) and new_vol > 0 and new_vol < best_volume:
                            best_volume = new_vol
                            best_positions = new_pos
            except Exception:
                pass
    
    # Final polish with Nelder-Mead on the best position
    if time.time() < deadline - 30:
        try:
            def objective3(pos_flat):
                if time.time() > deadline - 10:
                    return 1e10
                pos = pos_flat.reshape(n_tubes, 2)
                if not np.all(np.isfinite(pos)):
                    return 1e10
                vol = _compute_volume_fast(pos)
                if not np.isfinite(vol) or vol <= 0:
                    return 1e10
                return vol
            
            res = minimize(
                objective3,
                best_positions.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < best_volume:
                new_pos = res.x.reshape(n_tubes, 2)
                if np.all(np.isfinite(new_pos)):
                    new_vol = _compute_volume_fast(new_pos)
                    if np.isfinite(new_vol) and new_vol > 0 and new_vol < best_volume:
                        best_volume = new_vol
                        best_positions = new_pos
        except Exception:
            pass
    
    # Validate final result
    final_vol = _compute_volume_fast(best_positions)
    if not np.isfinite(final_vol) or final_vol <= 0:
        # Fall back to initial
        best_positions = positions_init
    
    return best_positions
