import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Strategy: Use a smooth log-sum-exp surrogate to enable gradient-based optimization
    (L-BFGS-B) with multiple restarts, followed by fine-grid polishing.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)

    # ---- Keich seed generation ----
    def keich_seed():
        seed_positions = np.zeros(n, dtype=np.float64)
        for i in range(n):
            # Convert i to binary string of length k
            bits = np.array(list(str(bin(i))[2:]), dtype=np.int8)
            bits = np.pad(bits, (k - len(bits), 0), constant_values=0)
            # Calculate position based on bit weights
            # The specific formula from the Keich construction for n=2^k
            seed_positions[i] = np.sum((1.0 - j) * bits * (2.0 ** -j) / k)
        return seed_positions

    seed_positions = keich_seed()
    a = np.arange(1, n + 1) / n  # shape (64,)

    # Precompute y-grid for the objective function
    num_h_opt = 1000
    ys = (np.arange(num_h_opt) + 0.5) / num_h_opt
    ws = (1.0 - ys) / n

    def union_area_fast(x):
        """
        Vectorized union-area over the optimization y-grid.
        x: shape (64,)
        Returns scalar area estimate.
        """
        Ls = x[None, :] + a[None, :] * ys[:, None]  # (num_h_opt, 64)
        sorted_Ls = np.sort(Ls, axis=1)             # (num_h_opt, 64)
        diffs = np.diff(sorted_Ls, axis=1)          # (num_h_opt, 63)
        extensions = np.minimum(ws[:, None], diffs) # (num_h_opt, 63)
        union_lens = ws + extensions.sum(axis=1)    # (num_h_opt,)
        return union_lens.mean()

    def union_area_fine(x, num_h_fine=5000):
        """Higher-resolution area for final evaluation."""
        ys_f = (np.arange(num_h_fine) + 0.5) / num_h_fine
        ws_f = (1.0 - ys_f) / n
        Ls = x[None, :] + a[None, :] * ys_f[:, None]
        sorted_Ls = np.sort(Ls, axis=1)
        diffs = np.diff(sorted_Ls, axis=1)
        extensions = np.minimum(ws_f[:, None], diffs)
        union_lens = ws_f + extensions.sum(axis=1)
        return union_lens.mean()

    def smooth_union_area(x, temp=0.01):
        """
        Smooth surrogate for union area using log-sum-exp to approximate min.
        min(a, b) ≈ -temp * logsumexp(-a/temp, -b/temp)
        This makes the objective differentiable for L-BFGS-B.
        """
        Ls = x[None, :] + a[None, :] * ys[:, None]  # (num_h_opt, 64)
        sorted_Ls = np.sort(Ls, axis=1)             # (num_h_opt, 64)
        diffs = np.diff(sorted_Ls, axis=1)          # (num_h_opt, 63)
        
        # Approximate min(diffs, w) using soft-min
        # min(d, w) = -t * log(exp(-d/t) + exp(-w/t))
        # To avoid overflow, use logsumexp trick:
        # log(exp(-d/t) + exp(-w/t)) = max(-d/t, -w/t) + log(exp(-d/t - max) + exp(-w/t - max))
        # Let M = max(-d/t, -w/t) = -min(d, w)/t
        # Then soft_min = -t * (M + log(exp(-d/t - M) + exp(-w/t - M)))
        # Note: -d/t - M = -d/t + min(d,w)/t = (min(d,w) - d)/t <= 0
        # So exp terms are <= 1, no overflow.
        
        neg_diffs_over_t = -diffs / temp
        neg_ws_over_t = -ws[:, None] / temp
        
        # M = max(neg_diffs_over_t, neg_ws_over_t)
        M = np.maximum(neg_diffs_over_t, neg_ws_over_t)
        
        # log(exp(neg_diffs_over_t - M) + exp(neg_ws_over_t - M))
        # Since M is the max, one term is 0, the other is <= 0.
        term1 = np.exp(neg_diffs_over_t - M)
        term2 = np.exp(neg_ws_over_t - M)
        
        log_sum = np.log(term1 + term2)
        
        soft_mins = -temp * (M + log_sum)
        
        union_lens = ws + soft_mins.sum(axis=1)
        return union_lens.mean()

    # ---- Optimization ----
    start_time = time.time()
    max_time = 620.0  # Safety margin

    best_x = seed_positions.copy()
    best_area_fine = union_area_fine(best_x)

    np.random.seed(123)

    # Restart list: Keich seed + perturbations
    restarts = [seed_positions.copy()]
    for scale in [0.1, 0.05, 0.01]:
        restarts.append(seed_positions + np.random.normal(0, scale, n))
    # Random starts in a reasonable range
    for _ in range(5):
        restarts.append(np.random.uniform(-0.2, 1.2, n))

    # Use L-BFGS-B with the smooth surrogate
    for idx, x0 in enumerate(restarts):
        if time.time() - start_time > max_time - 60:
            break
        
        try:
            # L-BFGS-B with smooth objective
            res = minimize(smooth_union_area, x0, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-10, 
                                    'gtol': 1e-8, 'disp': False})
            
            # Evaluate on fine grid to check if it's actually better
            candidate_area = union_area_fine(res.x)
            if candidate_area < best_area_fine:
                best_area_fine = candidate_area
                best_x = res.x.copy()
                
        except Exception:
            pass

    # ---- Global search with differential evolution if time permits ----
    if time.time() - start_time < max_time - 120:
        try:
            bounds = [(-0.5, 1.5)] * n
            # Use a coarse grid for DE objective to speed up
            def de_objective(x):
                return union_area_fast(x)
            
            res_de = differential_evolution(de_objective, bounds, 
                                            seed=42, maxiter=100, 
                                            tol=1e-6, popsize=15,
                                            mutation=(0.3, 1.2),
                                            recombination=0.8,
                                            polish=False,
                                            init='sobol')
            
            if res_de.success:
                candidate_area = union_area_fine(res_de.x)
                if candidate_area < best_area_fine:
                    best_area_fine = candidate_area
                    best_x = res_de.x.copy()
        except Exception:
            pass

    # ---- Final Polish with L-BFGS-B on the exact fine grid ----
    if time.time() - start_time < max_time - 10:
        def objective_fine(x):
            return union_area_fine(x, num_h_fine=5000)

        try:
            # Use L-BFGS-B with a small step to polish
            res_fine = minimize(objective_fine, best_x, method='L-BFGS-B',
                                options={'maxiter': 100, 'ftol': 1e-12, 
                                         'gtol': 1e-8, 'disp': False})
            if res_fine.fun < best_area_fine:
                best_area_fine = res_fine.fun
                best_x = res_fine.x.copy()
        except Exception:
            pass

    # Ensure the result is valid and finite
    if not np.all(np.isfinite(best_x)):
        return seed_positions
        
    return np.array(best_x, dtype=np.float64)
