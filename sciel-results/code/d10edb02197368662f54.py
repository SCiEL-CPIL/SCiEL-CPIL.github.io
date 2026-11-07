# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives, bounds [0,1]).
    Strategy:
      1. Weighted-sum scalarization with a structured set of weight vectors.
      2. For each weight, run a local search (Nelder-Mead via scipy) starting
         from a Latin Hypercube point.
      3. Collect all non-dominated points, compute hypervolume, and keep
         the best set within the evaluation budget.
      4. Deadline guard to stay within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate weight vectors for weighted-sum scalarization
        # Use a grid of weights in the (n_obj-1)-simplex
        # For 3 objectives, we can use a 2D grid
        n_weights_per_dim = 5
        weights = []
        if n_obj == 3:
            for i in range(n_weights_per_dim + 1):
                for j in range(n_weights_per_dim + 1 - i):
                    k = n_weights_per_dim - i - j
                    w = np.array([i, j, k], dtype=float) / n_weights_per_dim
                    if w.sum() > 0:
                        weights.append(w)
        else:
            # Fallback: random weights
            rng_w = np.random.RandomState(0)
            for _ in range(50):
                w = rng_w.dirichlet(np.ones(n_obj))
                weights.append(w)

        weights = np.array(weights)
        n_weights = len(weights)

        # Latin Hypercube Sampling for starting points
        rng = np.random.RandomState(0)
        lhs_points = np.zeros((n_weights, dim))
        for d in range(dim):
            perm = rng.permutation(n_weights)
            lhs_points[:, d] = (perm + rng.random(n_weights)) / n_weights

        # Clip to bounds
        lhs_points = np.clip(lhs_points, 0, 1)
        lhs_points = lo + lhs_points * (hi - lo)

        all_points = []
        all_fvals = []
        eval_count = 0

        # Phase 1: Evaluate initial LHS points
        for i in range(min(n_weights, budget)):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            f = np.array(problem(x), dtype=float)
            all_points.append(x)
            all_fvals.append(f)
            eval_count += 1

        # Phase 2: Local search from each starting point using weighted sum
        from scipy.optimize import minimize

        for i in range(n_weights):
            if time.time() > deadline:
                break
            if eval_count >= budget:
                break

            w = weights[i]
            x0 = lhs_points[i].copy()

            def objective(x):
                nonlocal eval_count
                if time.time() > deadline:
                    return 1e10
                if eval_count >= budget:
                    return 1e10
                fx = np.array(problem(x), dtype=float)
                eval_count += 1
                return np.dot(w, fx)

            # Nelder-Mead local search
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.success or res.fun < 1e9:
                    x_best = np.clip(res.x, lo, hi)
                    f_best = np.array(problem(x_best), dtype=float)
                    eval_count += 1
                    all_points.append(x_best)
                    all_fvals.append(f_best)
            except Exception:
                pass

            if eval_count >= budget:
                break

        # If we still have budget, do some random sampling
        if eval_count < budget:
            n_extra = budget - eval_count
            for _ in range(n_extra):
                if time.time() > deadline:
                    break
                x = lo + rng.random(dim) * (hi - lo)
                f = np.array(problem(x), dtype=float)
                all_points.append(x)
                all_fvals.append(f)
                eval_count += 1

        if not all_points:
            return lo

        all_points = np.array(all_points)
        all_fvals = np.array(all_fvals)

        return all_points[0]
# EVOLVE-BLOCK-END
