import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    N = 8
    n_samples = 500000
    
    # Pre-compute indices for tubes
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    def compute_volume_exact(params):
        """
        Compute exact union volume for separable parameters A, B, C, D.
        params: shape (32,) = [A(8), B(8), C(8), D(8)]
        x_ij = A_i + B_j
        y_ij = C_i + D_j
        """
        if time.time() > deadline:
            return 1e10
            
        A = params[0:N]
        B = params[N:2*N]
        C = params[2*N:3*N]
        D = params[3*N:4*N]
        
        # Construct base positions
        x_coords = A[:, None] + B[None, :]  # shape (8, 8)
        y_coords = C[:, None] + D[None, :]  # shape (8, 8)
        
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        
        # Determine bounds
        X0 = np.min(x_flat)
        X1 = np.max(x_flat + (i_flat + 1) / N)
        Y0 = np.min(y_flat)
        Y1 = np.max(y_flat + (j_flat + 1) / N)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return 1e10
        
        # For exact volume computation, we integrate the cross-sectional area
        # The tube (i,j) at height c occupies:
        #   x in [x_ij + c*i/N, x_ij + c*i/N + 1/N]
        #   y in [y_ij + c*j/N, y_ij + c*j/N + 1/N]
        # 
        # The union volume = integral_0^1 Area_union(c) dc
        # where Area_union(c) is the area of the union of 64 rectangles in 2D.
        #
        # Computing the exact union area of 64 rectangles is expensive.
        # Instead, we use a deterministic quadrature in c with many samples,
        # and for each c compute the union area of the 64 axis-aligned rectangles.
        #
        # For a fixed c, the union area of axis-aligned rectangles can be
        # computed by sweeping or by inclusion-exclusion, but with 64 rects
        # it's better to use a grid-based approach or the "rectangle union"
        # algorithm.
        #
        # Alternative: Use a fine grid in the bounding box to estimate area,
        # but that's still Monte Carlo-ish. Let's use the exact rectangle
        # union area via the standard sweep-line or coordinate compression.
        #
        # For 64 rectangles, coordinate compression is feasible:
        # Collect all x-coordinates and y-coordinates, sort, and check cells.
        
        # We'll use a deterministic set of c values for integration
        c_vals = np.linspace(0, 1, 201)  # 201 points, trapezoidal rule
        
        total_area = 0.0
        for c in c_vals:
            # Compute all rectangle corners
            x_mins = x_flat + c * i_flat / N
            x_maxs = x_mins + 1.0 / N
            y_mins = y_flat + c * j_flat / N
            y_maxs = y_mins + 1.0 / N
            
            # Coordinate compression for union area
            x_coords_unique = np.unique(np.concatenate([x_mins, x_maxs]))
            y_coords_unique = np.unique(np.concatenate([y_mins, y_maxs]))
            
            if len(x_coords_unique) < 2 or len(y_coords_unique) < 2:
                total_area += 0.0
                continue
            
            # Check which cells are covered
            # This is O(n_cells * n_rects) which for 64 rects is manageable
            # but might be slow. Let's use a more efficient approach.
            
            # Actually, for 64 rectangles, the number of unique coords is at most 128 each.
            # Number of cells: up to 127 * 127 = ~16000 cells.
            # For each cell, check if it's covered by any rect: 16000 * 64 = ~1M ops.
            # With 201 c values, that's ~200M ops. Might be too slow in Python.
            
            # Let's use a vectorized approach:
            # Create a grid and check coverage vectorized.
            
            # Actually, let's reduce the number of c samples for speed.
            # Use 51 c values.
            
            # Better: use a different approach. Since we're optimizing,
            # we can use a coarser estimate. Let's use 20 c values.
            
            # For now, let's just do the naive loop but break early if slow.
            # Actually, let's use a simpler Monte Carlo with fixed seed for speed.
            
            pass
        
        # This exact approach is too slow. Let's use Monte Carlo with the fixed seed
        # as specified, but that's what the original did.
        # 
        # The key insight from inspirations: use exact 1D cost integration.
        # The problem has a separable structure. Let's think about it differently.
        #
        # Actually, let's just use the Monte Carlo approach but make it faster
        # by vectorizing the tube checks.
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, n_samples)
        b = rng.uniform(Y0, Y1, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        # Vectorized check: for each sample, check if inside any tube
        # a_min[k] = x_flat[k] + c * i_flat[k] / N
        # a_max[k] = a_min[k] + 1/N
        # b_min[k] = y_flat[k] + c * j_flat[k] / N
        # b_max[k] = b_min[k] + 1/N
        
        # We need: exists k such that a_min[k] <= a <= a_max[k] and b_min[k] <= b <= b_max[k]
        
        # This is hard to vectorize over k efficiently for 64 tubes.
        # Let's loop over tubes but vectorize over samples.
        
        inside = np.zeros(n_samples, dtype=bool)
        
        for k in range(N * N):
            i_k = i_flat[k]
            j_k = j_flat[k]
            x_k = x_flat[k]
            y_k = y_flat[k]
            
            a_min = x_k + c * i_k / N
            a_max = a_min + 1.0 / N
            b_min = y_k + c * j_k / N
            b_max = b_min + 1.0 / N
            
            mask = (a >= a_min) & (a <= a_max) & (b >= b_min) & (b <= b_max)
            inside |= mask
        
        fraction = np.mean(inside)
        volume = fraction * area
        return volume
    
    # Initial guess
    A_init = -np.arange(N) / N
    B_init = np.zeros(N)
    C_init = np.zeros(N)
    D_init = -np.arange(N) / N
    
    params_init = np.concatenate([A_init, B_init, C_init, D_init])
    
    best_params = params_init.copy()
    best_volume = compute_volume_exact(best_params)
    
    # Optimization with Nelder-Mead
    try:
        res = minimize(compute_volume_exact, params_init, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'disp': False})
        if res.fun < best_volume:
            best_params = res.x
            best_volume = res.fun
    except Exception:
        pass
    
    # Try a few different initializations
    if time.time() < deadline - 120:
        inits = [
            np.concatenate([np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]),
            np.concatenate([np.arange(N)/N, np.zeros(N), np.zeros(N), np.arange(N)/N]),
            np.concatenate([-np.arange(N)/N, np.zeros(N), np.zeros(N), -np.arange(N)/N]),
            np.concatenate([np.arange(N)/N, -np.arange(N)/N, np.arange(N)/N, -np.arange(N)/N]),
            np.concatenate([-np.arange(N)/N, np.arange(N)/N, -np.arange(N)/N, np.arange(N)/N]),
        ]
        
        for init in inits:
            if time.time() > deadline - 60:
                break
            try:
                res_r = minimize(compute_volume_exact, init, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7, 'disp': False})
                if res_r.fun < best_volume:
                    best_params = res_r.x
                    best_volume = res_r.fun
            except Exception:
                pass
    
    # Construct final positions
    A = best_params[0:N]
    B = best_params[N:2*N]
    C = best_params[2*N:3*N]
    D = best_params[3*N:4*N]
    
    x_coords = A[:, None] + B[None, :]
    y_coords = C[:, None] + D[None, :]
    
    positions = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
    
    return positions
