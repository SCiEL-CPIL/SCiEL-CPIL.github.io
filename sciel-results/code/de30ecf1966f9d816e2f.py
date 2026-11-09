"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start local search: scatter starts, refine each with Nelder-Mead on weighted sums."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Phase 1: scatter initial points across the box
        n_starts = min(30, self.budget // 4)
        starts = lo + (hi - lo) * rng.random((n_starts, dim))

        # Phase 2: for each start, run local optimization with a weighted-sum objective
        # Use diverse weight vectors to cover the Pareto front
        # Generate weight vectors on the simplex (sum to 1, all >= 0)
        weight_vectors = []
        for _ in range(n_starts):
            w = rng.random(n_obj)
            w /= w.sum()
            weight_vectors.append(w)
        weight_vectors = np.array(weight_vectors)

        evaluated = []  # list of (x, f) pairs

        for i in range(n_starts):
            if time.time() > deadline:
                break
            if len(evaluated) >= self.budget:
                break

            x0 = starts[i]
            w = weight_vectors[i]

            # Define scalarized objective
            def scalarized(x):
                fx = problem(x)
                return float(np.dot(w, fx))

            # Nelder-Mead with tight maxfev to stay within budget
            maxfev = max(20, (self.budget - len(evaluated)) // (n_starts - i))
            maxfev = min(maxfev, 80)  # cap per-start evaluations

            res = minimize(
                scalarized,
                x0,
                method='Nelder-Mead',
                bounds=list(zip(lo, hi)),
                options={'maxiter': maxfev, 'maxfev': maxfev, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )

            x_best = np.clip(res.x, lo, hi)
            f_best = problem(x_best)
            evaluated.append((x_best, f_best))

            # Also record the starting point if it wasn't already recorded
            f_start = problem(starts[i])
            evaluated.append((starts[i], f_start))

            if time.time() > deadline:
                break

        # Phase 3: if we have remaining budget, do a few more random starts with local search
        while len(evaluated) < self.budget and time.time() < deadline:
            x0 = lo + (hi - lo) * rng.random(dim)
            w = rng.random(n_obj)
            w /= w.sum()

            def scalarized(x):
                fx = problem(x)
                return float(np.dot(w, fx))

            maxfev = max(10, (self.budget - len(evaluated)) // 2)
            maxfev = min(maxfev, 40)

            res = minimize(
                scalarized,
                x0,
                method='Nelder-Mead',
                bounds=list(zip(lo, hi)),
                options={'maxiter': maxfev, 'maxfev': maxfev, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )

            x_best = np.clip(res.x, lo, hi)
            f_best = problem(x_best)
            evaluated.append((x_best, f_best))

            f_start = problem(x0)
            if len(evaluated) < self.budget:
                evaluated.append((x0, f_start))

        return evaluated
# EVOLVE-BLOCK-END
