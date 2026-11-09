"""Z-order space-filling curve global search + CMA-ES local refinement for BBOB f4."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Z-order curve global scan followed by CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _z_order_point(self, idx, dim, lo, hi):
        """Convert a Z-order index to a point in [lo, hi]^dim.
        Uses Morton/Z-order: interleave bits of each coordinate."""
        # Number of bits per coordinate
        # We want roughly uniform coverage; use 4 bits per dim for 5 dims = 20 bits = ~1M points
        # But we only use a fraction. Let's use 3 bits per dim for 5 dims = 15 bits = 32768 points
        bits_per_dim = 3
        n_per_dim = 2 ** bits_per_dim  # 8
        # Extract bits for each dimension from the index
        coords = np.zeros(dim)
        for d in range(dim):
            # Get bits for dimension d: they are the low bits of (idx >> (d * bits_per_dim))
            bits = (idx >> (d * bits_per_dim)) & (n_per_dim - 1)
            # Map [0, n_per_dim-1] to [lo, hi]
            coords[d] = lo + (hi - lo) * bits / (n_per_dim - 1)
        return coords

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        lo = -5.0
        hi = 5.0

        calls_used = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        # Phase 1: Z-order space-filling curve scan for dense global coverage
        # Use a portion of the budget for global exploration
        # 3 bits per dim, 5 dims => 32768 possible points; use ~40% of budget
        n_scan = min(int(budget * 0.4), 32768)
        
        # We'll sample sequentially along the Z-order curve
        for idx in range(n_scan):
            if calls_used >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            pt = self._z_order_point(idx, dim, lo, hi)
            f_val = func(pt)
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # Phase 2: CMA-ES local refinement from the best point found
        remaining = budget - calls_used
        if remaining > 20:
            # Use scipy's CMA-ES via minimize
            # Set maxiter to use most of remaining budget
            # CMA-ES uses ~ (dim+2)*ln(dim+3)*ln(ln(dim+3)) per generation roughly
            # For dim=5, that's about 7*2.9*0.76 ≈ 15 per gen
            # So remaining/15 generations
            max_iter = max(1, int(remaining / 12))
            
            # Define wrapper that tracks calls
            cma_calls = [0]
            def cma_func(x):
                x = np.clip(x, lo, hi)
                cma_calls[0] += 1
                return func(x)
            
            try:
                res = minimize(
                    cma_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter * 3,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True,
                    }
                )
                calls_used += cma_calls[0]
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, lo, hi)
            except Exception:
                pass

            # Phase 3: If budget remains, do a second CMA-ES pass with tighter bounds
            remaining2 = budget - calls_used
            if remaining2 > 20:
                cma_calls2 = [0]
                def cma_func2(x):
                    x = np.clip(x, lo, hi)
                    cma_calls2[0] += 1
                    return func(x)
                try:
                    res2 = minimize(
                        cma_func2,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max(1, int(remaining2 / 10)),
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True,
                        }
                    )
                    calls_used += cma_calls2[0]
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = np.clip(res2.x, lo, hi)
                except Exception:
                    pass

        # Phase 4: Fine coordinate descent if budget remains
        remaining3 = budget - calls_used
        if remaining3 > 10 and time.time() < deadline - 5.0:
            fine_steps = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
            x = best_x.copy()
            fx = best_f
            for step in fine_steps:
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                for d in range(dim):
                    if calls_used >= budget or time.time() > deadline - 5.0:
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget or time.time() > deadline - 5.0:
                            break
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, lo, hi)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new

            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
