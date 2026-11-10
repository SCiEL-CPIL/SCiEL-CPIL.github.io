import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    
    N = 8
    w = 1.0 / N
    shifts = np.arange(N) / N  # [0, 1/8, ..., 7/8]
    
    def union_length(x, c):
        starts = x + c * shifts
        ends = starts + w
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        total = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for i in range(1, N):
            if s_sorted[i] <= cur_end:
                if e_sorted[i] > cur_end:
                    cur_end = e_sorted[i]
            else:
                total += cur_end - cur_start
                cur_start = s_sorted[i]
                cur_end = e_sorted[i]
        total += cur_end - cur_start
        return total
    
    def cost_func(x, n_samples):
        c_vals = np.linspace(0.0, 1.0, n_samples)
        L_vals = np.empty(n_samples)
        for idx in range(n_samples):
            L_vals[idx] = union_length(x, c_vals[idx])
        return np.trapz(L_vals ** 2, c_vals)
    
    def fallback_positions():
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
        return np.stack([keich[i_flat], keich[j_flat]], axis=1)
    
    # Initial guess
    x0 = -np.arange(N) * w / 2
    
    best_x = x0.copy()
    best_cost = cost_func(x0, 501)
    
    rng = np.random.default_rng(42)
    starts = [x0.copy()]
    for _ in range(15):
        starts.append(rng.uniform(-0.3, 0.3, size=N))
    
    for x_start in starts:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(lambda v: cost_func(v, 501), x_start, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 30:
        try:
            res = minimize(lambda v: cost_func(v, 2001), best_x, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
        positions = fallback_positions()
    
    return positions
