"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start optimizer for BBOB-style f19 (Griewank-Rosenbrock).
    
    Combines model-based initialization along the Rosenbrock valley chain
    with multi-start L-BFGS-B and golden-section coordinate refinement.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Track function calls to enforce budget
        call_count = 0
        best_x = [None]
        best_f = [np.inf]

        def time_ok():
            return (time.time() < deadline) and (call_count < budget)

        def counted_func(x):
            nonlocal call_count
            if not time_ok():
                return np.inf
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            call_count += 1
            try:
                fv = float(func(x))
            except Exception:
                fv = np.inf
            if np.isfinite(fv) and fv < best_f[0]:
                best_f[0] = fv
                best_x[0] = x.copy()
            return fv

        def local_lbfgs(x0, max_evals=None, tight=False):
            """Run L-BFGS-B from x0, updating global best."""
            if not time_ok():
                return
            if max_evals is None:
                max_evals = min(budget - call_count, 80)
            if max_evals < 5:
                return
            
            bounds = [(-5, 5)] * n
            t_start = time.time()
            
            def callback(xk):
                return (time.time() - t_start > 60) or (call_count >= budget)
            
            try:
                if tight:
                    res = minimize(counted_func, x0, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 500, 'maxfun': max_evals,
                                            'gtol': 1e-12, 'ftol': 1e-16,
                                            'maxls': 50})
                else:
                    res = minimize(counted_func, x0, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 200, 'maxfun': max_evals,
                                            'gtol': 1e-10, 'ftol': 1e-14,
                                            'maxls': 40})
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = res.x.copy()
            except Exception:
                pass

        # Phase 1: Model-based initialization along Rosenbrock valley chain
        # The Griewank-Rosenbrock minimum lies near the chain where
        # x_i ≈ 2 * x_{i-1}^2 with x_0 near 1
        chain_starts = []
        for x0_val in [0.8, 1.0, 1.2, -1.0, -0.8]:
            x = np.zeros(n)
            x[0] = x0_val
            for i in range(1, n):
                x[i] = 2.0 * x[i-1] * x[i-1]
                # Clip to box to keep in bounds
                x[i] = np.clip(x[i], -5.0, 5.0)
            chain_starts.append(x.copy())
        
        # Also add the origin and a few diverse points
        chain_starts.append(np.zeros(n))
        chain_starts.append(np.ones(n) * 0.5)
        chain_starts.append(np.ones(n) * -0.5)
        
        # Evaluate chain starts and run L-BFGS-B from the best few
        chain_results = []
        for x0 in chain_starts:
            if not time_ok():
                break
            fv = counted_func(x0)
            chain_results.append((fv, x0.copy()))
        
        # Sort by function value and run L-BFGS-B from top candidates
        chain_results.sort(key=lambda t: t[0])
        for fv, x0 in chain_results[:4]:
            if not time_ok():
                break
            local_lbfgs(x0, max_evals=60)

        # Phase 2: Additional diverse starting points with L-BFGS-B
        if time_ok():
            # Stratified random points
            n_extra = 6
            for i in range(n_extra):
                if not time_ok():
                    break
                x = np.array([(i + rng.random()) / n_extra * 10 - 5 for _ in range(n)])
                x = np.clip(x, -5.0, 5.0)
                fv = counted_func(x)
                if fv < best_f[0]:
                    local_lbfgs(x, max_evals=50)
            
            # Random points for robustness
            for _ in range(3):
                if not time_ok():
                    break
                x = rng.uniform(-5, 5, size=n)
                fv = counted_func(x)
                if fv < best_f[0] * 0.5:  # Only refine if significantly better
                    local_lbfgs(x, max_evals=40)

        # Phase 3: Final polish with tighter L-BFGS-B
        if best_x[0] is not None and time_ok():
            remaining = budget - call_count
            if remaining > 20:
                local_lbfgs(best_x[0], max_evals=min(remaining, 40), tight=True)

        # Phase 4: Coordinate-wise golden-section search for final precision
        if best_x[0] is not None and time_ok():
            remaining = budget - call_count
            if remaining > 30:
                x_curr = best_x[0].copy()
                f_curr = best_f[0]
                gr = (1.0 + np.sqrt(5.0)) / 2.0
                inv_gr = 1.0 / gr
                inv_gr2 = inv_gr * inv_gr

                # Multiple passes of coordinate-wise search
                for pass_num in range(3):
                    if not time_ok() or (budget - call_count < 15):
                        break

                    for dim_idx in range(n):
                        if not time_ok() or (budget - call_count < 5):
                            break

                        # Adaptive step: start moderate, shrink if no improvement
                        step = 0.3
                        improved = False
                        for attempt in range(2):
                            if not time_ok() or (budget - call_count < 5):
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

                            # Do a few more iterations of golden-section
                            for _ in range(10):
                                if not time_ok() or (budget - call_count < 3):
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

                        if f_curr < best_f[0]:
                            best_f[0] = f_curr
                            best_x[0] = x_curr.copy()

        # Phase 5: Final high-precision polish pass
        if best_x[0] is not None and time_ok():
            remaining = budget - call_count
            if remaining > 10:
                # Run a final tight L-BFGS-B from the current best
                local_lbfgs(best_x[0], max_evals=min(remaining, 30), tight=True)

                # If we still have budget, do one more coordinate pass with smaller steps
                if time_ok() and (budget - call_count > 15):
                    x_curr = best_x[0].copy()
                    f_curr = best_f[0]
                    gr = (1.0 + np.sqrt(5.0)) / 2.0
                    inv_gr = 1.0 / gr
                    inv_gr2 = inv_gr * inv_gr

                    for dim_idx in range(n):
                        if not time_ok() or (budget - call_count < 5):
                            break

                        step = 0.1
                        lo = max(-5.0, x_curr[dim_idx] - step)
                        hi = min(5.0, x_curr[dim_idx] + step)
                        if hi - lo < 1e-12:
                            continue

                        x1 = x_curr.copy()
                        x2 = x_curr.copy()
                        x1[dim_idx] = lo + inv_gr2 * (hi - lo)
                        x2[dim_idx] = lo + inv_gr * (hi - lo)

                        f1 = counted_func(x1)
                        f2 = counted_func(x2)

                        for _ in range(8):
                            if not time_ok() or (budget - call_count < 3):
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

                        best_1d_x = x_curr.copy()
                        best_1d_f = f_curr
                        if f1 < best_1d_f:
                            best_1d_f = f1
                            best_1d_x[dim_idx] = x1[dim_idx]
                        if f2 < best_1d_f:
                            best_1d_f = f2
                            best_1d_x[dim_idx] = x2[dim_idx]

                        if best_1d_f < f_curr - 1e-14:
                            x_curr = best_1d_x
                            f_curr = best_1d_f

                    if f_curr < best_f[0]:
                        best_f[0] = f_curr
                        best_x[0] = x_curr.copy()

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
