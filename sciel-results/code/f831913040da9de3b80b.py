"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer combining weighted-sum decomposition with
    evolutionary refinement and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Store all evaluated points and their objective vectors
        X = []  # list of x vectors
        F = []  # list of f vectors
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X.append(x)
            F.append(f)
            used += 1
            return f

        # Phase 1: Random sampling to establish initial diversity
        n_random = max(10, self.budget // 4)
        for i in range(n_random):
            if time.time() > deadline or used >= self.budget:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            eval_point(x)

        # Helper: find non-dominated points
        def get_non_dominated():
            if len(F) == 0:
                return []
            F_arr = np.array(F)
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return [i for i in range(n) if not dominated[i]]

        # Phase 2: Weighted-sum decomposition with L-BFGS-B
        # Generate diverse weight vectors
        n_weights = 12
        weights = []
        # Add extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
        # Add uniform random weights
        for _ in range(n_weights - self.n_obj):
            w = rng.random(self.n_obj)
            w /= np.sum(w)
            weights.append(w)
        weights = np.array(weights)

        for w in weights:
            if time.time() > deadline or used >= self.budget:
                break

            # Choose best starting point based on current weighted sum
            if len(F) > 0:
                F_arr = np.array(F)
                weighted_sums = F_arr @ w
                best_idx = np.argmin(weighted_sums)
                x0 = X[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0

            # Objective function for L-BFGS-B
            def obj(x):
                nonlocal used
                if used >= self.budget:
                    return 0.0
                x = np.clip(x, lo, hi)
                f = np.array(problem(x), dtype=float)
                X.append(x)
                F.append(f)
                used += 1
                return np.sum(f * w)

            try:
                result = minimize(
                    obj, x0, method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 40, 'maxls': 20, 'maxfun': 30}
                )
            except Exception:
                pass

        # Phase 3: Evolutionary refinement with adaptive mutation
        if used < self.budget and time.time() < deadline:
            # Build population from non-dominated points
            nd_idx = get_non_dominated()
            if len(nd_idx) == 0:
                nd_idx = list(range(len(F)))

            # Main evolutionary loop
            pop_size = max(8, min(20, (self.budget - used) // 6))
            iteration = 0
            max_iterations = 50

            while used < self.budget and time.time() < deadline and iteration < max_iterations:
                iteration += 1

                # Re-compute non-dominated set
                nd_idx = get_non_dominated()
                if len(nd_idx) < 2:
                    nd_idx = list(range(len(F)))

                # Adaptive mutation scale: starts large, decays over time
                progress = iteration / max_iterations
                mutation_scale = 0.3 * (1 - 0.7 * progress) * (hi - lo)

                # Pick two parents from non-dominated set
                p1 = nd_idx[rng.integers(len(nd_idx))]
                p2 = nd_idx[rng.integers(len(nd_idx))]

                # Crossover: blend parents
                x1 = X[p1]
                x2 = X[p2]
                alpha = rng.random(self.dim)
                child = alpha * x1 + (1 - alpha) * x2

                # Mutation
                child += mutation_scale * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = eval_point(child)
                if fc is None:
                    break

                # Occasionally try a local refinement around a good point
                if rng.random() < 0.3 and used < self.budget and time.time() < deadline:
                    # Pick a random non-dominated point for local search
                    if len(nd_idx) > 0:
                        local_idx = nd_idx[rng.integers(len(nd_idx))]
                        x0 = X[local_idx].copy()

                        # Try optimizing each objective separately
                        for obj_i in range(self.n_obj):
                            if time.time() > deadline or used >= self.budget:
                                break

                            def local_obj(x):
                                nonlocal used
                                if used >= self.budget:
                                    return 0.0
                                x = np.clip(x, lo, hi)
                                f = np.array(problem(x), dtype=float)
                                X.append(x)
                                F.append(f)
                                used += 1
                                return f[obj_i]

                            try:
                                minimize(
                                    local_obj, x0, method='L-BFGS-B',
                                    bounds=list(zip(lo, hi)),
                                    options={'maxiter': 15, 'maxfun': 20}
                                )
                            except Exception:
                                pass

        return X
# EVOLVE-BLOCK-END
