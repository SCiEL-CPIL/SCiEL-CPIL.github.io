# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_2d_union_area_exact(x, y, c, N=8):
    """Compute the exact 2D union area of N^2 axis-aligned squares at height c.
    Uses sweep line / rectangle union algorithm.
    """
    i_idx = np.arange(N)
    
    # For each tube (i, j), the square is:
    # x: [x_i + c*i/N, x_i + c*i/N + 1/N]
    # y: [y_j + c*j/N, y_j + c*j/N + 1/N]
    
    # Collect all rectangles
    rects = []
    for i in range(N):
        for j in range(N):
            x0 = x[i] + c * i / N
            x1 = x0 + 1.0 / N
            y0 = y[j] + c * j / N
            y1 = y0 + 1.0 / N
            rects.append((x0, x1, y0, y1))
    
    if not rects:
        return 0.0
    
    # Use sweep line with coordinate compression
    # Collect all unique x and y coordinates
    xs = sorted(set([r[0] for r in rects] + [r[1] for r in rects]))
    ys = sorted(set([r[2] for r in rects] + [r[3] for r in rects]))
    
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    
    # Create index maps
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    
    # Grid of cells
    grid = np.zeros((len(ys) - 1, len(xs) - 1), dtype=np.int8)
    
    for (x0, x1, y0, y1) in rects:
        xi0 = x_idx[x0]
        xi1 = x_idx[x1]
        yi0 = y_idx[y0]
        yi1 = y_idx[y1]
        grid[yi0:yi1, xi0:xi1] = 1
    
    # Compute area
    dx = np.diff(xs)
    dy = np.diff(ys)
    cell_areas = np.outer(dy, dx)
    area = np.sum(grid * cell_areas)
    
    return area

def _compute_3d_volume_exact(x, y, N=8, K=200):
    """Compute the 3D union volume by integrating exact 2D union area over c in [0, 1]."""
    total_vol = 0.0
    for k in range(K):
        c = k / K
        total_vol += _compute_2d_union_area_exact(x, y, c, N)
    return total_vol / K

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600.0

    # Initial candidates for x and y coordinates
    # Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # Symmetric
    symmetric = -np.arange(N, dtype=np.float64) / (2 * N)
    
    # Zero
    origin = np.zeros(N, dtype=np.float64)
    
    # Linear
    linear = np.linspace(0, 1 - 1 / N, N, dtype=np.float64)
    
    # Random perturbation of keich
    rng = np.random.default_rng(42)
    keich_perturbed = keich + rng.uniform(-0.05, 0.05, N)
    
    candidates = [keich, symmetric, origin, linear, keich_perturbed]
    
    best_params = np.concatenate([keich, keich])
    best_vol = float('inf')
    
    # Evaluate initial candidates
    for x in candidates:
        for y in candidates:
            vol = _compute_3d_volume_exact(x, y, N, K=50)
            if vol < best_vol:
                best_vol = vol
                best_params = np.concatenate([x, y])
                
    # Optimization
    if time.time() < deadline - 120.0:
        from scipy.optimize import minimize
        
        def objective(params):
            x = params[:N]
            y = params[N:]
            return _compute_3d_volume_exact(x, y, N, K=100)
            
        x0 = best_params.copy()
        
        # Powell method is good for this
        try:
            res = minimize(objective, x0, method='Powell',
                          options={'maxiter': 300, 'ftol': 1e-10, 'disp': False})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
        # Try Nelder-Mead if time remains
        if time.time() < deadline - 60.0:
            try:
                res2 = minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
                if res2.fun < best_vol:
                    best_vol = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass

    # Final high-precision evaluation and potential polish
    if time.time() < deadline - 30.0:
        try:
            final_vol = _compute_3d_volume_exact(best_params[:N], best_params[N:], N, K=200)
            if final_vol < best_vol:
                best_vol = final_vol
        except Exception:
            pass

    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    best_x = best_params[:N]
    best_y = best_params[N:]
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    return positions

# EVOLVE-BLOCK-END
