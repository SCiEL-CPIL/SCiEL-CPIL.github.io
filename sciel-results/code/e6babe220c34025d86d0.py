"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB-style f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Track function calls to enforce budget
        call_count = 0

        def counted_func(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        # Generate diverse starting points
        # 1. Origin
        # 2. Stratified random points (LHS-like)
        # 3. Points near boundaries
        n_starts = 8
        starts = []
        starts.append(np.zeros(n))
        
        # Add stratified points
        for i in range(n_starts - 1):
            # Stratified sampling in each dimension
            x = np.array([(i + rng.random()) / n_starts * 10 - 5 for _ in range(n)])
            starts.append(x)
            
        # Add a few random points for robustness
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, size=n))

        best_x = None
        best_f = np.inf
        bounds = [(-5, 5)] * n

        for x0 in starts:
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            # Limit evaluations per run to allow multiple restarts
            # L-BFGS-B needs ~50-80 evals for good convergence in 5D
            remaining = budget - call_count
            max_evals_per_run = min(remaining, 60)
            if max_evals_per_run < 10:
                break

            try:
                res = minimize(counted_func, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'maxfun': max_evals_per_run,
                                        'gtol': 1e-8, 'ftol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Final polish: coordinate-wise golden-section search
        if best_x is not None and call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            if remaining > 10:
                # First try L-BFGS-B with tighter tolerance
                try:
                    res = minimize(counted_func, best_x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 200, 'maxfun': min(remaining, 40),
                                            'gtol': 1e-10, 'ftol': 1e-14})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

            # Coordinate-wise golden-section search for final precision
            if best_x is not None and call_count < budget and time.time() < deadline:
                remaining = budget - call_count
                if remaining > 20:
                    x_curr = best_x.copy()
                    f_curr = best_f
                    # Golden ratio
                    gr = (1.0 + np.sqrt(5.0)) / 2.0
                    inv_gr = 1.0 / gr
                    inv_gr2 = inv_gr * inv_gr

                    # Multiple passes of coordinate-wise search
                    for pass_num in range(5):
                        if call_count >= budget or time.time() > deadline:
                            break
                        if budget - call_count < 10:
                            break

                        for dim_idx in range(n):
                            if call_count >= budget or time.time() > deadline:
                                break
                            if budget - call_count < 4:
                                break

                            # Adaptive step: start moderate, shrink if no improvement
                            step = 0.5
                            improved = False
                            for attempt in range(3):
                                if call_count >= budget or time.time() > deadline:
                                    break
                                if budget - call_count < 4:
                                    break

                                lo = max(-5.0, x_curr[dim_idx] - step)
                                hi = min(5.0, x_curr[dim_idx] + step)
                                if hi - lo < 1e-12:
                                    break

                                # Golden-section search in [lo, hi] for this dimension
                                x1 = x_curr.copy()
                                x2 = x_curr.copy()
                                x1[dim_idx] = lo + inv_gr2 * (hi - lo)
                                x2[dim_idx] = lo + inv_gr * (hi - lo)

                                f1 = counted_func(x1)
                                f2 = counted_func(x2)

                                # Check if center is better
                                f_center = f_curr  # x_curr value
                                # Do a few more iterations of golden-section
                                for _ in range(15):
                                    if call_count >= budget or time.time() > deadline:
                                        break
                                    if f1 < f2:
                                        hi = x2[dim_idx]
                                        x2 = x1.copy()
                                        f2 = f1
                                        x1[dim_idx] = lo + inv_gr2 * (hi - lo)
                                        f1 = counted_func(x1)
                                    else:
                                        lo = x1[dim_idx]
                                        x1 = x2.copy()
                                        f1 = f2
                                        x2[dim_idx] = lo + inv_gr * (hi - lo)
                                        f2 = counted_func(x2)

                                # Best point in this 1D search
                                best_1d_x = x_curr.copy()
                                best_1d_f = f_curr
                                if f1 < best_1d_f:
                                    best_1d_f = f1
                                    best_1d_x[dim_idx] = x1[dim_idx]
                                if f2 < best_1d_f:
                                    best_1d_f = f2
                                    best_1d_x[dim_idx] = x2[dim_idx]

                                if best_1d_f < f_curr - 1e-12:
                                    x_curr = best_1d_x
                                    f_curr = best_1d_f
                                    improved = True
                                    break
                                else:
                                    step *= 0.5

                            if f_curr < best_f:
                                best_f = f_curr
                                best_x = x_curr.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
