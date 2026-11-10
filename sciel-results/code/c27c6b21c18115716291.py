import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)

    # Keich seed
    seed_positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bits = np.array(list(str(bin(i))[2:]), dtype=np.int8)
        bits = np.pad(bits, (k - len(bits), 0), constant_values=0)
        seed_positions[i] = np.sum((1.0 - j) * bits * (2.0 ** -j) / k)

    a = np.arange(1, n + 1) / n

    # Coarse grid for fast optimization
    num_h_coarse = 1000
    ys_c = (np.arange(num_h_coarse) + 0.5) / num_h_coarse
    ws_c = (1.0 - ys_c) / n

    # Fine grid for final evaluation
    num_h_fine = 20000
    ys_f = (np.arange(num_h_fine) + 0.5) / num_h_fine
    ws_f = (1.0 - ys_f) / n

    def union_area(x, ys, ws):
        Ls = x[None, :] + a[None, :] * ys[:, None]
        sorted_Ls = np.sort(Ls, axis=1)
        diffs = np.diff(sorted_Ls, axis=1)
        extensions = np.minimum(ws[:, None], diffs)
        union_lens = ws + extensions.sum(axis=1)
        return union_lens.mean()

    def objective_coarse(x):
        return union_area(x, ys_c, ws_c)

    def objective_fine(x):
        return union_area(x, ys_f, ws_f)

    start_time = time.time()
    max_time = 625.0

    best_x = seed_positions.copy()
    best_area_c = objective_coarse(best_x)

    np.random.seed(42)

    # Generate diverse restarts
    restarts = [seed_positions.copy()]
    # Small perturbations around seed
    for _ in range(12):
        restarts.append(seed_positions + np.random.normal(0, 0.02, n))
    # Medium perturbations
    for _ in range(8):
        restarts.append(seed_positions + np.random.normal(0, 0.05, n))
    # Random uniform
    for _ in range(6):
        restarts.append(np.random.uniform(-0.3, 1.3, n))
    # Structured: shift by small amounts
    for shift in [0.01, -0.01, 0.02, -0.02]:
        restarts.append(seed_positions + shift)

    # L-BFGS-B optimization on coarse grid
    for idx, x0 in enumerate(restarts):
        if time.time() - start_time > max_time - 20:
            break
        try:
            res = minimize(objective_coarse, x0, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
            if res.fun < best_area_c:
                best_area_c = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Coordinate descent polish on coarse grid
    if time.time() - start_time < max_time - 15:
        x = best_x.copy()
        step = 0.01
        for outer in range(300):
            if time.time() - start_time > max_time - 10:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > max_time - 10:
                    break
                cur = objective_coarse(x)
                best_delta = 0.0
                best_val = cur
                for delta in (step, -step, step * 0.5, -step * 0.5, step * 0.25, -step * 0.25,
                              step * 2, -step * 2):
                    x[i] += delta
                    val = objective_coarse(x)
                    x[i] -= delta
                    if val < best_val - 1e-15:
                        best_val = val
                        best_delta = delta
                if best_delta != 0.0:
                    x[i] += best_delta
                    improved = True
            if not improved:
                break
            step *= 0.8

        if objective_coarse(x) < best_area_c:
            best_area_c = objective_coarse(x)
            best_x = x.copy()

    # Final fine-grid L-BFGS-B polish
    if time.time() - start_time < max_time - 5:
        try:
            res_fine = minimize(objective_fine, best_x, method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res_fine.fun < objective_fine(best_x):
                best_x = res_fine.x.copy()
        except Exception:
            pass

    # One more coordinate descent on fine grid if time permits
    if time.time() - start_time < max_time - 2:
        x = best_x.copy()
        step = 0.005
        for outer in range(50):
            if time.time() - start_time > max_time - 1:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > max_time - 1:
                    break
                cur = objective_fine(x)
                best_delta = 0.0
                best_val = cur
                for delta in (step, -step, step * 0.5, -step * 0.5):
                    x[i] += delta
                    val = objective_fine(x)
                    x[i] -= delta
                    if val < best_val - 1e-15:
                        best_val = val
                        best_delta = delta
                if best_delta != 0.0:
                    x[i] += best_delta
                    improved = True
            if not improved:
                break
            step *= 0.7

        if objective_fine(x) < objective_fine(best_x):
            best_x = x.copy()

    return np.array(best_x, dtype=np.float64)
