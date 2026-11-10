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
    2. A smooth log-sum-exp surrogate for the union area to enable efficient gradient-based optimization.
    3. L-BFGS-B local refinement on the smooth surrogate.
    4. A final exact-metric polish phase with coordinate descent and L-BFGS-B on the true piecewise-linear objective.
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

    # Precompute y-grid for the smooth surrogate (coarse for speed during main optimization)
    num_h_smooth = 200
    ys_smooth = (np.arange(num_h_smooth) + 0.5) / num_h_smooth
    ws_smooth = (1.0 - ys_smooth) / n

    def union_area_exact(x, num_h=2000):
        """
        Exact union-area over a y-grid.
        x: shape (64,)
        Returns scalar area estimate (mean union length over y).
        """
        ys = (np.arange(num_h) + 0.5) / num_h
        ws = (1.0 - ys) / n
        Ls = x[None, :] + a[None, :] * ys[:, None]
        sorted_Ls = np.sort(Ls, axis=1)
        diffs = np.diff(sorted_Ls, axis=1)
        extensions = np.minimum(ws[:, None], diffs)
        union_lens = ws + extensions.sum(axis=1)
        return union_lens.mean()

    def union_area_smooth(x, beta=100.0):
        """
        Smooth surrogate for union area using log-sum-exp to approximate min(L_j+1 - L_j, w).
        The union length at height y is: w + sum_k min(d_k, w) where d_k = L_{(k+1)} - L_{(k)}.
        We approximate min(d, w) = -max(w-d, 0) = -softplus(w-d, beta) ... wait, that's not quite right.
        Actually, min(d, w) is a concave function of d. We want a smooth approximation.
        min(d, w) ≈ - (1/beta) * log(1 + exp(-beta*(d - w))) + w * (1 - sigmoid(beta*(d-w)))? 
        Simpler: min(d, w) = (d + w - |d - w|) / 2.
        |d - w| ≈ (1/beta) * log(1 + exp(2*beta*|d-w|/2))? No, softplus approximation for |z| is (1/beta)*log(1+exp(2*beta*z)) - z for z>0... 
        Let's use: min(d, w) ≈ w - softplus(w - d, beta) where softplus(z) = log(1+exp(z))/beta... no.
        Standard softplus: sp(z) = log(1+exp(z)). Then min(d,w) = d - max(d-w, 0) = d - sp(d-w).
        So min(d, w) ≈ d - (1/beta)*sp(beta*(d-w)).
        This is smooth and differentiable.
        """
        Ls = x[None, :] + a[None, :] * ys_smooth[:, None]
        sorted_Ls = np.sort(Ls, axis=1)
        diffs = np.diff(sorted_Ls, axis=1)
        
        # min(d, w) ≈ d - softplus(d - w) / beta
        # To avoid overflow in exp, we can use a stable softplus.
        # softplus(z) = max(z, 0) + log(1 + exp(-|z|))
        def softplus(z):
            return np.where(z > 20, z, np.log1p(np.exp(z)))
        
        z = beta * (diffs - ws_smooth[:, None])
        sp_z = softplus(z) / beta
        min_dw = diffs - sp_z
        
        # Clip to be non-negative just in case of numerical issues, though mathematically it should be >= 0
        min_dw = np.maximum(min_dw, 0)
        
        union_lens = ws_smooth + min_dw.sum(axis=1)
        return union_lens.mean()

    def objective_smooth(x):
        return union_area_smooth(x, beta=50.0)

    # ---- Optimization with time guard ----
    start_time = time.time()
    max_time = 620.0  # leave 10 s safety margin

    best_x = seed_positions.copy()
    best_area_exact = union_area_exact(best_x, num_h=500)

    np.random.seed(42)

    # Restart list: seed + perturbations
    restarts = [seed_positions.copy()]
    for _ in range(10):
        restarts.append(seed_positions + np.random.normal(0, 0.05, n))
    for _ in range(5):
        restarts.append(np.random.uniform(-0.5, 1.5, n))

    # Phase 1: Optimize smooth surrogate
    for idx, x0 in enumerate(restarts):
        if time.time() - start_time > max_time - 60:
            break
        try:
            res = minimize(objective_smooth, x0, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
            # Evaluate exact area for the candidate
            exact_area = union_area_exact(res.x, num_h=500)
            if exact_area < best_area_exact:
                best_area_exact = exact_area
                best_x = res.x
        except Exception:
            pass

    # Phase 2: Polish with exact metric using L-BFGS-B (it will struggle with kinks but can still move)
    if time.time() - start_time < max_time - 20:
        def objective_exact_fast(x):
            return union_area_exact(x, num_h=1000)
        
        try:
            res_fine = minimize(objective_exact_fast, best_x, method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False})
            if res_fine.fun < best_area_exact:
                best_area_exact = res_fine.fun
                best_x = res_fine.x
        except Exception:
            pass

    # Phase 3: Coordinate descent on exact metric
    if time.time() - start_time < max_time - 5:
        x = best_x.copy()
        step = 0.01
        for _ in range(100):
            if time.time() - start_time > max_time - 2:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > max_time - 2:
                    break
                best_val = union_area_exact(x, num_h=500)
                best_delta = 0.0
                for delta in (step, -step, step/2, -step/2, step/4, -step/4):
                    x[i] += delta
                    val = union_area_exact(x, num_h=500)
                    x[i] -= delta
                    if val < best_val - 1e-14:
                        best_val = val
                        best_delta = delta
                if best_delta != 0.0:
                    x[i] += best_delta
                    improved = True
            if not improved:
                break
            step *= 0.8
        final_area = union_area_exact(x, num_h=500)
        if final_area < best_area_exact:
            best_area_exact = final_area
            best_x = x

    return np.array(best_x, dtype=np.float64)
