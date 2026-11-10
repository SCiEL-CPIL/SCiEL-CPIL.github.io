import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 595.0
    
    N = 8
    
    def compute_cost(X):
        # X: array of shape (N,)
        # Returns integral of L(c)^2 from 0 to 1
        
        # Critical points
        criticals = [0.0, 1.0]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # c(i-j)/N = X_j - X_i + delta
                # delta is 0, 1/N, -1/N
                for delta in [0.0, 1.0/N, -1.0/N]:
                    c_val = N * (X[j] - X[i] + delta) / (i - j)
                    if 0.0 <= c_val <= 1.0:
                        criticals.append(c_val)
        
        criticals = sorted(set(criticals))
        
        total_cost = 0.0
        
        def get_length(c):
            starts = X + c * np.arange(N) / N
            ends = starts + 1.0 / N
            order = np.argsort(starts)
            s_sorted = starts[order]
            e_sorted = ends[order]
            
            length = 0.0
            current_end = -np.inf
            for s, e in zip(s_sorted, e_sorted):
                if s > current_end:
                    length += (e - s)
                    current_end = e
                else:
                    if e > current_end:
                        length += (e - current_end)
                        current_end = e
            return length

        for k in range(len(criticals) - 1):
            c_start = criticals[k]
            c_end = criticals[k+1]
            if c_end - c_start < 1e-12:
                continue
            
            L_start = get_length(c_start)
            L_end = get_length(c_end)
            
            A = (L_end - L_start) / (c_end - c_start)
            B = L_start - A * c_start
            
            # Integrate (A c + B)^2
            # F(c) = A^2 c^3/3 + A B c^2 + B^2 c
            F_start = (A**2 * c_start**3) / 3.0 + (A * B * c_start**2) + (B**2 * c_start)
            F_end = (A**2 * c_end**3) / 3.0 + (A * B * c_end**2) + (B**2 * c_end)
            
            total_cost += F_end - F_start
            
        return total_cost

    # Keich initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # Try to optimize
    best_x = keich
    best_cost = compute_cost(keich)
    
    # Try Nelder-Mead from Keich
    try:
        res = minimize(compute_cost, keich, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_cost:
            best_cost = res.fun
            best_x = res.x
    except Exception:
        pass
        
    # Try a few random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        x0 = rng.uniform(-0.5, 0.5, size=N)
        try:
            res = minimize(compute_cost, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
         # Fallback to keich if optimization failed badly
         base_x = keich[i_flat]
         base_y = keich[j_flat]
         positions = np.stack([base_x, base_y], axis=1)
         
    return positions
