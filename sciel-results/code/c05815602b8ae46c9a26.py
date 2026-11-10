import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 620.0

    # Keich binary construction for initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Scale and shift for better starting point
    X0 = keich * 0.5 + 0.25
    Y0 = keich * 0.5 + 0.25

    # Precompute indices for vectorization
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    def compute_1d_union_length(positions, c, N):
        """Compute total length of union of intervals [pos[i] + c*i/N, pos[i] + c*i/N + 1/N] for i=0..N-1"""
        starts = positions + c * np.arange(N) / N
        ends = starts + 1.0 / N
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        # Merge intervals
        total = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] <= cur_end:
                cur_end = max(cur_end, e_sorted[k])
            else:
                total += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
        total += cur_end - cur_start
        return total

    def exact_volume(X, Y, N):
        """
        Compute the exact integral of Lx(c)*Ly(c) dc using adaptive quadrature
        on the break points where the union structure changes.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        def get_breaks(f):
            breaks = {0.0, 1.0}
            invN = 1.0 / N
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    for offset in [0.0, invN, -invN]:
                        denom = (i - j)
                        if denom != 0:
                            c_val = (f[j] - f[i] + offset) * N / denom
                            if 0.0 < c_val < 1.0:
                                breaks.add(c_val)
            return sorted(breaks)
        
        x_breaks = get_breaks(X)
        y_breaks = get_breaks(Y)
        all_breaks = sorted(set(x_breaks + y_breaks))
        
        total_vol = 0.0
        for k in range(len(all_breaks) - 1):
            c_mid = (all_breaks[k] + all_breaks[k+1]) / 2.0
            Lx = compute_1d_union_length(X, c_mid, N)
            Ly = compute_1d_union_length(Y, c_mid, N)
            total_vol += Lx * Ly * (all_breaks[k+1] - all_breaks[k])
            
        return total_vol

    def objective(params):
        if time.time() > deadline - 5:
            return 1e9
        X = params[:N]
        Y = params[N:]
        return exact_volume(X, Y, N)
    
    initial_params = np.concatenate([X0, Y0])
    
    best_params = initial_params.copy()
    best_score = objective(initial_params)
    
    # Local search with Nelder-Mead
    # We use a moderate number of restarts. The exact volume is piecewise polynomial,
    # so Nelder-Mead should be able to find local minima effectively.
    for restart in range(8):
        if time.time() > deadline - 60:
            break
            
        if restart == 0:
            start = initial_params.copy()
        else:
            # Add noise based on restart index for diversity
            rng_noise = np.random.default_rng(restart * 100 + 1)
            noise = rng_noise.normal(0, 0.05, size=2*N)
            start = initial_params + noise
        
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 30:
            break
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res_final = minimize(objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'disp': False})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
            
    X_opt = best_params[:N]
    Y_opt = best_params[N:]
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
