# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_2d_union_area_exact(x, y, c, N=8):
    """Compute the exact 2D union area of N^2 axis-aligned squares at height c.
    Uses coordinate compression for exact rectangle union.
    """
    i_idx = np.arange(N)
    
    # For each tube (i, j), the square is:
    # x: [x_i + c*i/N, x_i + c*i/N + 1/N]
    # y: [y_j + c*j/N, y_j + c*j/N + 1/N]
    
    # Collect all rectangles
    x_starts = x + c * i_idx / N
    x_ends = x_starts + 1.0 / N
    y_starts = y + c * i_idx / N
    y_ends = y_starts + 1.0 / N
    
    # Get all unique x and y coordinates
    all_x = np.concatenate([x_starts, x_ends])
    all_y = np.concatenate([y_starts, y_ends])
    
    xs = np.unique(all_x)
    ys = np.unique(all_y)
    
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    
    # Create index maps
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    
    # Grid of cells
    grid = np.zeros((len(ys) - 1, len(xs) - 1), dtype=np.int8)
    
    for i in range(N):
        for j in range(N):
            sx = x[i] + c * i / N
            ex = sx + 1.0 / N
            sy = y[j] + c * j / N
            ey = sy + 1.0 / N
            
            xi0 = x_idx[sx]
            xi1 = x_idx[ex]
            yi0 = y_idx[sy]
            yi1 = y_idx[ey]
            
            if xi0 < xi1 and yi0 < yi1:
                grid[yi0:yi1, xi0:xi1] = 1
    
    # Compute area
    dx = np.diff(xs)
    dy = np.diff(ys)
    cell_areas = np.outer(dy, dx)
    area = np.sum(grid * cell_areas)
    
    return area

def _compute_3d_volume(x, y, N=8, K=50):
    """Compute the 3D union volume by integrating exact 2D union area over c in [0, 1]."""
    total_vol = 0.0
    for k in range(K):
        c = k / K
        total_vol += _compute_2d_union_area_exact(x, y, c, N)
    return total_vol / K

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 570.0  # Safety margin

    # Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    symmetric = -np.arange(N, dtype=np.float64) / (2 * N)
    origin = np.zeros(N, dtype=np.float64)
    linear = np.linspace(0, 1 - 1/N, N, dtype=np.float64)
    
    rng = np.random.default_rng(42)
    
    # Generate candidate x and y positions
    candidates = [keich, symmetric, origin, linear]
    for _ in range(6):
        candidates.append(keich + rng.uniform(-0.1, 0.1, N))
        candidates.append(rng.uniform(-0.3, 0.3, N))
    
    best_x = keich.copy()
    best_y = keich.copy()
    best_vol = float('inf')
    
    # Evaluate initial candidates
    for x in candidates:
        for y in candidates:
            vol = _compute_3d_volume(x, y, N, K=30)
            if vol < best_vol:
                best_vol = vol
                best_x = x.copy()
                best_y = y.copy()
    
    from scipy.optimize import minimize
    
    # Define objective function for optimization
    def objective(params):
        x = params[:N]
        y = params[N:]
        return _compute_3d_volume(x, y, N, K=40)
    
    # Nelder-Mead optimization
    if time.time() < deadline - 200.0:
        try:
            x0 = np.concatenate([best_x, best_y])
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_vol:
                best_vol = res.fun
                best_x = res.x[:N].copy()
                best_y = res.x[N:].copy()
        except Exception:
            pass
    
    # L-BFGS-B for smoother convergence
    if time.time() < deadline - 120.0:
        try:
            x0 = np.concatenate([best_x, best_y])
            res2 = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-14})
            if res2.fun < best_vol:
                best_vol = res2.fun
                best_x = res2.x[:N].copy()
                best_y = res2.x[N:].copy()
        except Exception:
            pass
    
    # Polish with higher precision
    if time.time() < deadline - 60.0:
        def objective_polish(params):
            x = params[:N]
            y = params[N:]
            return _compute_3d_volume(x, y, N, K=100)
        
        try:
            x0 = np.concatenate([best_x, best_y])
            res3 = minimize(objective_polish, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res3.fun < best_vol:
                best_vol = res3.fun
                best_x = res3.x[:N].copy()
                best_y = res3.x[N:].copy()
        except Exception:
            pass
    
    # Additional random restarts if time permits
    if time.time() < deadline - 30.0:
        for _ in range(3):
            if time.time() > deadline - 15.0:
                break
            r_x = keich + rng.uniform(-0.05, 0.05, N)
            r_y = keich + rng.uniform(-0.05, 0.05, N)
            try:
                x0 = np.concatenate([r_x, r_y])
                res_r = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res_r.fun < best_vol:
                    best_vol = res_r.fun
                    best_x = res_r.x[:N].copy()
                    best_y = res_r.x[N:].copy()
            except Exception:
                pass
    
    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
