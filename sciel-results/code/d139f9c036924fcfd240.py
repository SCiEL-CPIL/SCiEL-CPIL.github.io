import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0 # 600s limit, keep margin
    
    N = 8
    # Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    def objective(params):
        Z = np.zeros(N)
        Z[0] = 0.0
        Z[1:] = params
        
        # Compute 1D volume integral
        # Use 100 points for speed, should be accurate enough
        c_vals = np.linspace(0, 1, 100)
        lengths = np.zeros_like(c_vals)
        
        for idx, c in enumerate(c_vals):
            starts = Z + c * np.arange(N) / N
            ends = starts + 1.0 / N
            order = np.argsort(starts)
            s_sorted = starts[order]
            e_sorted = ends[order]
            
            total_len = 0.0
            current_end = -np.inf
            for k in range(N):
                s = s_sorted[k]
                e = e_sorted[k]
                if s > current_end:
                    total_len += e - s
                    current_end = e
                else:
                    if e > current_end:
                        total_len += e - current_end
                        current_end = e
            lengths[idx] = total_len
            
        vol_1d = np.trapz(lengths, c_vals)
        return vol_1d * vol_1d
        
    # Initial guess
    x0 = keich[1:] - keich[0]
    
    # Optimize
    # Use multiple restarts if time permits
    best_res = None
    best_obj = np.inf
    
    for restart in range(5):
        if time.time() > deadline - 60:
            break
            
        if restart == 0:
            x_start = x0
        else:
            # Add noise
            x_start = x0 + np.random.normal(0, 0.1, size=N-1)
            
        res = minimize(objective, x_start, method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000})
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_res = res
            
    if best_res is None:
        Z_opt = keich
    else:
        Z_opt = np.zeros(N)
        Z_opt[0] = 0.0
        Z_opt[1:] = best_res.x
        
    # Construct 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = Z_opt[i_flat]
    base_y = Z_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
