import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    n_tubes = N * N
    
    i_idx = np.repeat(np.arange(N), N)
    j_idx = np.tile(np.arange(N), N)
    
    def objective(params):
        X = params[:N]
        Y = params[N:2*N]
        
        x_base = X[i_idx]
        y_base = Y[j_idx]
        
        X0 = np.min(x_base)
        X1 = np.max(x_base + (i_idx + 1) / N)
        Y0 = np.min(y_base)
        Y1 = np.max(y_base + (j_idx + 1) / N)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return 1e10
            
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, 500000)
        b = rng.uniform(Y0, Y1, 500000)
        c = rng.uniform(0, 1, 500000)
        
        x_tube_start = x_base[None, :] + c[:, None] * i_idx[None, :] / N
        y_tube_start = y_base[None, :] + c[:, None] * j_idx[None, :] / N
        
        x_tube_end = x_tube_start + 1.0 / N
        y_tube_end = y_tube_start + 1.0 / N
        
        inside_x = (a[:, None] >= x_tube_start) & (a[:, None] <= x_tube_end)
        inside_y = (b[:, None] >= y_tube_start) & (b[:, None] <= y_tube_end)
        
        inside = inside_x & inside_y
        is_inside = np.any(inside, axis=1)
        
        fraction = np.mean(is_inside)
        V = fraction * area
        
        return V

    # Starting points
    # 1. Star
    X_star = -np.arange(N) / (2.0 * N)
    Y_star = -np.arange(N) / (2.0 * N)
    starts = [np.concatenate([X_star, Y_star])]
    
    # 2. Zero
    X_zero = np.zeros(N)
    Y_zero = np.zeros(N)
    starts.append(np.concatenate([X_zero, Y_zero]))
    
    # 3. Random
    rng_init = np.random.default_rng(42)
    for _ in range(3):
        X_rand = rng_init.uniform(-1, 1, N)
        Y_rand = rng_init.uniform(-1, 1, N)
        starts.append(np.concatenate([X_rand, Y_rand]))
    
    best_V = float('inf')
    best_params = starts[0]
    
    for x0 in starts:
        try:
            result = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'maxfev': 5000})
            if result.fun < best_V:
                best_V = result.fun
                best_params = result.x
        except Exception:
            continue
            
    X_opt = best_params[:N]
    Y_opt = best_params[N:2*N]
    
    x_base = X_opt[i_idx]
    y_base = Y_opt[j_idx]
    positions = np.stack([x_base, y_base], axis=1)
    
    return positions
