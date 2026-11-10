import numpy as np
import time
from scipy.optimize import minimize

def _compute_volume(positions, N=8, n_samples=500000, seed=12345):
    """
    Compute the Monte Carlo volume estimate for given positions.
    positions: (64, 2) array.
    """
    rng = np.random.default_rng(seed)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x_starts = positions[:, 0]
    y_starts = positions[:, 1]
    
    x_ends = x_starts + (i_flat + 1) / N
    y_ends = y_starts + (j_flat + 1) / N
    
    X0 = np.min(x_starts)
    X1 = np.max(x_ends)
    Y0 = np.min(y_starts)
    Y1 = np.max(y_ends)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')
    
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Vectorized tube checks
    x_lo = x_starts[np.newaxis, :] + c[:, np.newaxis] * i_flat[np.newaxis, :] / N
    x_hi = x_lo + 1.0 / N
    
    y_lo = y_starts[np.newaxis, :] + c[:, np.newaxis] * j_flat[np.newaxis, :] / N
    y_hi = y_lo + 1.0 / N
    
    inside_x = (a[:, np.newaxis] >= x_lo) & (a[:, np.newaxis] <= x_hi)
    inside_y = (b[:, np.newaxis] >= y_lo) & (b[:, np.newaxis] <= y_hi)
    
    inside_any = np.any(inside_x & inside_y, axis=1)
    
    frac = np.mean(inside_any)
    return frac * (X1 - X0) * (Y1 - Y0)

def _keich_1d(N=8):
    """Compute Keich binary construction for 1D positions."""
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
    deadline = start_time + 600  # 30s margin for safety
    
    # Initial guess: Keich construction
    base_x_1d = _keich_1d(N)
    base_y_1d = _keich_1d(N)
    
    x0 = base_x_1d.copy()
    y0 = base_y_1d.copy()
    initial_params = np.concatenate([x0, y0])
    
    def objective(params):
        x_1d = params[:N]
        y_1d = params[N:]
        
        x_grid = np.tile(x_1d[:, np.newaxis], (1, N))
        y_grid = np.tile(y_1d[np.newaxis, :], (N, 1))
        
        positions = np.stack([x_grid.flatten(), y_grid.flatten()], axis=1)
        
        # Use a smaller sample count for optimization to speed up, 
        # but keep it high enough for stability.
        # 50000 is ~10x faster than 500000.
        V = _compute_volume(positions, N=N, n_samples=50000)
        return V
    
    best_params = initial_params.copy()
    best_val = objective(best_params)
    
    # Optimization with time guard
    try:
        # Use Nelder-Mead with a smaller maxiter, relying on the speed of the new objective
        # We check time in the callback if possible, or just limit iterations.
        # Since each eval is fast (~0.05s), 1000 iterations * 17 evals/iter ~ 17000 evals * 0.05s = 850s.
        # That's too many. Let's reduce maxiter or use a tighter tol.
        # Or use Powell which might converge faster.
        
        # Let's try Nelder-Mead with maxiter=200.
        # 200 * 17 * 0.05 = 170s. Safe.
        
        res = minimize(objective, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if time.time() < deadline:
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
                
            # If we have time, do a second pass with more iterations
            if time.time() < deadline - 30:
                res2 = minimize(objective, best_params, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x
                    
    except Exception:
        pass

    # Final verification with full sample count (optional, but good for score accuracy if time permits)
    # The evaluator will run its own MC with 500k samples. 
    # We just need to return the positions.
    
    x_1d = best_params[:N]
    y_1d = best_params[N:]
    x_grid = np.tile(x_1d[:, np.newaxis], (1, N))
    y_grid = np.tile(y_1d[np.newaxis, :], (N, 1))
    final_positions = np.stack([x_grid.flatten(), y_grid.flatten()], axis=1)
    
    return final_positions
