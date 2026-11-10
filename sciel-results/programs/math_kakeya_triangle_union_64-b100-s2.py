import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    This implementation uses:
    1. The Keich construction as seed.
    2. A vectorized union-area evaluation (vectorized over y) for speed.
    3. L-BFGS-B local refinement with multiple restarts.
    4. A coordinate-descent polishing step to escape shallow local minima.
    5. A strict wall-clock guard so the function returns within 630 s.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)

    # ---- Keich seed ----
    seed_positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bits = np.array(list(str(bin(i))[2:]), dtype=np.int8)
        bits = np.pad(bits, (k - len(bits), 0), constant_values=0)
        seed_positions[i] = np.sum((1.0 - j) * bits * (2.0 ** -j) / k)

    a = np.arange(1, n + 1) / n  # shape (64,)

    # Precompute y-grid and w for a coarse objective (used during optimization)
    # Use 500 heights for a good trade-off between fidelity and speed.
    num_h = 500
    ys = (np.arange(num_h) + 0.5) / num_h          # shape (500,)
    ws = (1.0 - ys) / n                             # shape (500,)

    def union_area_fast(x):
        """
        Vectorized union-area over the coarse y-grid.
        x: shape (64,)
        Returns scalar area estimate (mean union length over y).
        """
        # Ls[y, i] = x[i] + a[i] * ys[y]  ->  shape (500, 64)
        Ls = x[None, :] + a[None, :] * ys[:, None]  # (500, 64)
        # Sort along axis=1
        sorted_Ls = np.sort(Ls, axis=1)             # (500, 64)
        diffs = np.diff(sorted_Ls, axis=1)          # (500, 63)
        # extensions[y, k] = min(ws[y], diffs[y, k])
        extensions = np.minimum(ws[:, None], diffs) # (500, 63)
        union_lens = ws + extensions.sum(axis=1)    # (500,)
        return union_lens.mean()

    def union_area_fine(x, num_h_fine=2000):
        """Higher-resolution area (used for final polishing checks)."""
        ys_f = (np.arange(num_h_fine) + 0.5) / num_h_fine
        ws_f = (1.0 - ys_f) / n
        Ls = x[None, :] + a[None, :] * ys_f[:, None]
        sorted_Ls = np.sort(Ls, axis=1)
        diffs = np.diff(sorted_Ls, axis=1)
        extensions = np.minimum(ws_f[:, None], diffs)
        union_lens = ws_f + extensions.sum(axis=1)
        return union_lens.mean()

    def objective(x):
        return union_area_fast(x)

    # ---- Optimization with time guard ----
    start_time = time.time()
    max_time = 620.0  # leave 10 s safety margin

    best_x = seed_positions.copy()
    best_area = objective(best_x)

    np.random.seed(42)

    # Restart list: seed + perturbations + random
    restarts = [seed_positions.copy()]
    for _ in range(8):
        restarts.append(seed_positions + np.random.normal(0, 0.03, n))
    for _ in range(6):
        restarts.append(np.random.uniform(-0.5, 1.5, n))

    for idx, x0 in enumerate(restarts):
        if time.time() - start_time > max_time - 15:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            if res.fun < best_area:
                best_area = res.fun
                best_x = res.x
        except Exception:
            pass

    # ---- Coordinate-descent polish on the coarse objective ----
    if time.time() - start_time < max_time - 10:
        x = best_x.copy()
        step = 0.02
        for _ in range(200):  # cap on outer iterations
            if time.time() - start_time > max_time - 5:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > max_time - 5:
                    break
                best_val = objective(x)
                best_delta = 0.0
                # try ± step, ± step/2, ± step/4
                for delta in (step, -step, step/2, -step/2, step/4, -step/4):
                    x[i] += delta
                    val = objective(x)
                    x[i] -= delta
                    if val < best_val - 1e-14:
                        best_val = val
                        best_delta = delta
                if best_delta != 0.0:
                    x[i] += best_delta
                    improved = True
            if not improved:
                break
            step *= 0.7
        if objective(x) < best_area:
            best_area = objective(x)
            best_x = x

    # ---- Final fine-grid polish using L-BFGS-B on the higher-resolution objective ----
    if time.time() - start_time < max_time - 5:
        def objective_fine(x):
            return union_area_fine(x, num_h_fine=2000)
        
        try:
            res_fine = minimize(objective_fine, best_x, method='L-BFGS-B',
                                options={'maxiter': 100, 'ftol': 1e-10, 'disp': False})
            if res_fine.fun < best_area:
                best_area = res_fine.fun
                best_x = res_fine.x
        except Exception:
            pass

    return np.array(best_x, dtype=np.float64)
