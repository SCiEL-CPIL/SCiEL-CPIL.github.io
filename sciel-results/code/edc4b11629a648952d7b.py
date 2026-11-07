"""Rewritten optimizer using Nelder-Mead + CMA-ES style adaptive search for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import scipy.stats


class Optimizer:
    """Adaptive derivative-free optimizer with Nelder-Mead and global restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        call_count = 0
        best_x = None
        best_f = np.inf
        all_results = []

        def safe_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                x = np.asarray(x, dtype=np.float64)
                val = func(x)
                call_count += 1
                if not np.isfinite(val):
                    return np.inf
                val = float(val)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                call_count += 1
                return np.inf

        def run_nelder_mead(x0, max_iter=200):
            """Run Nelder-Mead from x0, return (best_f, best_x, calls_used)."""
            nonlocal call_count
            local_calls = 0
            local_best_f = np.inf
            local_best_x = x0.copy()

            def obj(x):
                nonlocal local_calls, local_best_f, local_best_x
                if call_count >= budget or time.time() > deadline:
                    return np.inf
                local_calls += 1
                try:
                    val = func(np.asarray(x, dtype=np.float64))
                    call_count += 1
                    if not np.isfinite(val):
                        return np.inf
                    val = float(val)
                    if val < local_best_f:
                        local_best_f = val
                        local_best_x = np.asarray(x, dtype=np.float64).copy()
                    return val
                except Exception:
                    call_count += 1
                    return np.inf

            try:
                res = scipy.optimize.minimize(
                    obj, x0, method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
                )
                if np.isfinite(res.fun) and res.fun < local_best_f:
                    local_best_f = res.fun
                    local_best_x = res.x.copy()
            except Exception:
                pass

            return local_best_f, local_best_x, local_calls

        # Phase 1: Sobol sequence for global coverage
        num_global = min(60, budget // 3)
        if num_global < 10:
            num_global = 10

        try:
            sobol = scipy.stats.qmc.Sobol(d=d, scramble=True, seed=12345)
            k = int(np.ceil(np.log2(num_global)))
            num_points = 2 ** k
            if num_points > budget // 2:
                num_points = budget // 2
                k = int(np.ceil(np.log2(num_points)))
                if 2 ** k > num_points:
                    num_points = 2 ** (k - 1)

            points_unit = sobol.random(num_points)
            points = points_unit * 10.0 - 5.0

            for p in points:
                if call_count >= budget or time.time() > deadline:
                    break
                f = safe_call(p)
                if f < np.inf:
                    all_results.append((f, p.copy()))
        except Exception:
            rng = np.random.default_rng(42)
            for _ in range(num_global):
                if call_count >= budget or time.time() > deadline:
                    break
                p = rng.uniform(-5, 5, d)
                f = safe_call(p)
                if f < np.inf:
                    all_results.append((f, p.copy()))

        # Sort by fitness
        all_results.sort(key=lambda x: x[0])

        # Phase 2: Local refinement from top candidates
        top_k = min(8, len(all_results))
        for i in range(top_k):
            if call_count >= budget or time.time() > deadline:
                break

            x0 = all_results[i][1]
            f0, x_best, _ = run_nelder_mead(x0, max_iter=100)

            if f0 < best_f:
                best_f = f0
                best_x = x_best.copy()

        # Phase 3: Adaptive restarts with larger perturbations
        if best_x is not None and call_count < budget:
            rng = np.random.default_rng(777)
            remaining = budget - call_count

            # Use remaining budget for more restarts
            num_restarts = min(remaining // 30, 15)

            for r in range(num_restarts):
                if call_count >= budget or time.time() > deadline:
                    break

                # Adaptive perturbation size based on progress
                progress = 1.0 - (r / max(1, num_restarts))
                sigma = 0.5 + 2.0 * progress  # Larger early, smaller later

                if rng.random() < 0.4:
                    # Local perturbation around best
                    x0 = best_x + rng.normal(0, sigma, d)
                else:
                    # Global restart
                    x0 = rng.uniform(-5, 5, d)

                x0 = np.clip(x0, -5, 5)

                f0, x_best, _ = run_nelder_mead(x0, max_iter=50)

                if f0 < best_f:
                    best_f = f0
                    best_x = x_best.copy()

        # Phase 4: Final fine-tuning if budget allows
        if best_x is not None and call_count < budget and time.time() < deadline:
            try:
                def final_obj(x):
                    if call_count >= budget or time.time() > deadline:
                        return np.inf
                    try:
                        val = func(np.asarray(x, dtype=np.float64))
                        call_count += 1
                        if not np.isfinite(val):
                            return np.inf
                        return float(val)
                    except Exception:
                        call_count += 1
                        return np.inf

                res = scipy.optimize.minimize(
                    final_obj, best_x, method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
