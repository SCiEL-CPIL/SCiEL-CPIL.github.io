"""Optimizer for RE34 multi-objective problem using MOEA/D with L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """MOEA/D-based multi-objective optimizer for RE34 (5 vars, 3 obj)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Evaluation counter
        eval_count = 0

        def count_eval():
            nonlocal eval_count
            eval_count += 1

        def obj(x):
            """Wrapper that counts evaluations."""
            count_eval()
            return np.array(problem(x), dtype=np.float64)

        # Generate weight vectors for MOEA/D (3 objectives -> simplex)
        # Use a systematic grid on the 2-simplex
        n_points_per_edge = 6  # gives 21 weight vectors for 3 objectives
        weights = []
        for i in range(n_points_per_edge + 1):
            for j in range(n_points_per_edge + 1 - i):
                k = n_points_per_edge - i - j
                w = np.array([i, j, k], dtype=np.float64) / n_points_per_edge
                weights.append(w)
        weights = np.array(weights)
        n_sub = len(weights)

        # Initial population: one random point per subproblem
        rng = np.random.default_rng(42)
        X = np.zeros((n_sub, dim))
        F = np.zeros((n_sub, n_obj))

        for i in range(n_sub):
            x = lo + span * rng.random(dim)
            X[i] = x
            F[i] = obj(x)

        # L-BFGS-B refinement for each subproblem
        # Budget allocation: n_sub initial + n_sub * max_iter per subproblem
        # Each L-BFGS-B call uses multiple function evaluations internally
        # We need to be careful: scipy's minimize counts each function call
        # Let's use a small number of iterations per subproblem

        # Allocate remaining budget
        remaining = budget - eval_count
        # Each subproblem gets roughly remaining / n_sub evaluations
        # L-BFGS-B typically uses ~10-20 evals per iteration
        max_iter_per_sub = max(1, int(remaining // (n_sub * 15)))
        max_iter_per_sub = min(max_iter_per_sub, 10)  # cap to avoid overshoot

        for i in range(n_sub):
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break

            w = weights[i]
            x0 = X[i].copy()

            def weighted_sum(x, w=w):
                f = obj(x)
                return np.dot(f, w)

            try:
                res = minimize(
                    weighted_sum,
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': max_iter_per_sub, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success or res.fun < np.inf:
                    x_new = np.clip(res.x, lo, hi)
                    f_new = obj(x_new)
                    X[i] = x_new
                    F[i] = f_new
            except Exception:
                pass

        # Additional local search on non-dominated points
        # Find non-dominated set
        def is_non_dominated(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            return ~dominated

        nd_mask = is_non_dominated(F)
        nd_indices = np.where(nd_mask)[0]

        # Use remaining budget for additional L-BFGS-B runs from diverse starts
        remaining = budget - eval_count
        if remaining > 20:
            n_extra = min(len(nd_indices), max(1, remaining // 20))
            for idx in nd_indices[:n_extra]:
                if eval_count >= budget:
                    break
                if time.time() > deadline:
                    break

                w = weights[idx]
                x0 = X[idx].copy() + 0.05 * span * rng.standard_normal(dim)
                x0 = np.clip(x0, lo, hi)

                def weighted_sum2(x, w=w):
                    f = obj(x)
                    return np.dot(f, w)

                try:
                    res = minimize(
                        weighted_sum2,
                        x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 5, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res.success or res.fun < np.inf:
                        x_new = np.clip(res.x, lo, hi)
                        f_new = obj(x_new)
                        X[idx] = x_new
                        F[idx] = f_new
                except Exception:
                    pass

        # Final: ensure we don't exceed budget
        return X
