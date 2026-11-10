import numpy as np
from scipy.optimize import minimize
import time


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 590

    def compute_L(f, c):
        starts = f + c * np.arange(N) / N
        ends = f + 1.0 / N + c * np.arange(N) / N
        intervals = np.column_stack([starts, ends])
        intervals = intervals[np.argsort(intervals[:, 0])]

        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1] + 1e-12:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return sum(end - start for start, end in merged)

    def objective(f):
        c_vals = np.linspace(0, 1, 100)
        L_vals = np.array([compute_L(f, c) for c in c_vals])
        return np.trapz(L_vals ** 2, c_vals)

    # Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    best_f = keich.copy()
    best_obj = objective(keich)

    # Primary optimization
    result = minimize(objective, keich, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-11})
    if result.fun < best_obj:
        best_f = result.x
        best_obj = result.fun

    # Additional restarts if time allows
    if time.time() < deadline:
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline:
                break
            f_init = keich + rng.normal(0, 0.2, N)
            res = minimize(objective, f_init, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_obj:
                best_f = res.x
                best_obj = res.fun

    f_opt = best_f

    # Build positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = f_opt[i_flat]
    base_y = f_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions
