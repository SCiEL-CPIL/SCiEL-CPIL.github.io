"""Multi-objective optimizer for RE34 using MOEA/D decomposition with L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """MOEA/D-based multi-objective optimizer with weighted-sum decomposition
    and L-BFGS-B local search for systematic Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(123)

        all_X = []
        all_F = []
        eval_count = [0]

        def count_eval(x):
            if eval_count[0] >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count[0] += 1
            return f

        # ================================================================
        # Phase 1: Initial diverse sampling using stratified random design
        # ================================================================
        n_init = 30
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(self.dim)
            count_eval(x)

        # ================================================================
        # Phase 2: MOEA/D decomposition - 40 weight subproblems
        # ================================================================
        n_sub = 40
        weights = []

        # Add extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)

        # Add systematic weight vectors covering the simplex
        # For 3 objectives, use Dirichlet distribution samples
        for _ in range(n_sub - self.n_obj):
            w = rng.dirichlet(np.ones(self.n_obj))
            weights.append(w)

        weights = np.array(weights)

        # For each subproblem, run L-BFGS-B from a diverse starting point
        for i, w in enumerate(weights):
            if time.time() > deadline or eval_count[0] >= self.budget:
                break

            # Choose starting point: use best point for this weight from current archive
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                # Weighted distance to ideal (use min values)
                ideal = F_arr.min(axis=0)
                weighted_dist = np.abs(F_arr - ideal) @ w
                # Select from diverse starting points
                start_idx = rng.integers(len(F_arr))
                x0 = all_X[start_idx].copy()
            else:
                x0 = (lo + hi) / 2.0 + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                x0 = np.clip(x0, lo, hi)

            def obj_func(x, w=w):
                if eval_count[0] >= self.budget:
                    return np.inf
                fx = count_eval(x)
                if fx is None:
                    return np.inf
                return float(np.sum(fx * w))

            try:
                result = minimize(
                    obj_func, x0, method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 30, 'maxls': 10, 'ftol': 1e-8}
                )
                if result.fun < np.inf:
                    x_res = np.clip(result.x, lo, hi)
                    count_eval(x_res)
            except Exception:
                pass

        # ================================================================
        # Phase 3: Neighborhood-based local search refinement
        # ================================================================
        if eval_count[0] < self.budget and time.time() < deadline:
            # Get non-dominated set
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                X_arr = np.array(all_X)

                def get_non_dominated(F_arr):
                    n = len(F_arr)
                    if n == 0:
                        return []
                    dominated = np.zeros(n, dtype=bool)
                    for i in range(n):
                        if dominated[i]:
                            continue
                        for j in range(n):
                            if i == j or dominated[j]:
                                continue
                            if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                                dominated[i] = True
                                break
                            if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                                dominated[j] = True
                    return [i for i in range(n) if not dominated[i]]

                nd_idx = get_non_dominated(F_arr)

                # Refine each non-dominated point with multiple small perturbations
                n_refine = min(len(nd_idx), max(1, (self.budget - eval_count[0]) // 8))
                for idx in nd_idx[:n_refine]:
                    if time.time() > deadline or eval_count[0] >= self.budget:
                        break

                    x_nd = X_arr[idx]

                    # Try multiple random directions with decreasing step size
                    for attempt in range(4):
                        if eval_count[0] >= self.budget or time.time() > deadline:
                            break
                        step_size = 0.05 * (hi - lo) / (attempt + 1)
                        x_new = x_nd + step_size * rng.standard_normal(self.dim)
                        x_new = np.clip(x_new, lo, hi)
                        count_eval(x_new)

        # ================================================================
        # Phase 4: Final tight local polish on best points
        # ================================================================
        if eval_count[0] < self.budget and time.time() < deadline:
            if len(all_F) > 0:
                F_arr = np.array(all_F)

                def get_non_dominated(F_arr):
                    n = len(F_arr)
                    if n == 0:
                        return []
                    dominated = np.zeros(n, dtype=bool)
                    for i in range(n):
                        if dominated[i]:
                            continue
                        for j in range(n):
                            if i == j or dominated[j]:
                                continue
                            if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                                dominated[i] = True
                                break
                            if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                                dominated[j] = True
                    return [i for i in range(n) if not dominated[i]]

                nd_idx = get_non_dominated(F_arr)

                # Polish top non-dominated points with tight L-BFGS-B
                n_polish = min(5, len(nd_idx), max(1, (self.budget - eval_count[0]) // 6))
                for idx in nd_idx[:n_polish]:
                    if time.time() > deadline or eval_count[0] >= self.budget:
                        break

                    x_nd = all_X[idx]
                    f_nd = F_arr[idx]

                    # Use a weight vector pointing toward this point
                    # Find closest weight
                    F_arr_full = np.array(all_F)
                    ideal = F_arr_full.min(axis=0)
                    ant_ideal = F_arr_full.max(axis=0)
                    norm_f = (f_nd - ideal) / (ant_ideal - ideal + 1e-12)
                    norm_f = np.clip(norm_f, 0, 1)
                    w_polish = 1.0 / (norm_f + 1e-6)
                    w_polish /= w_polish.sum()

                    def obj_polish(x, w=w_polish):
                        if eval_count[0] >= self.budget:
                            return np.inf
                        fx = count_eval(x)
                        if fx is None:
                            return np.inf
                        return float(np.sum(fx * w))

                    try:
                        result = minimize(
                            obj_polish, x_nd, method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 20, 'maxls': 8, 'ftol': 1e-10}
                        )
                        if result.fun < np.inf:
                            x_res = np.clip(result.x, lo, hi)
                            count_eval(x_res)
                    except Exception:
                        pass

        return all_X
# EVOLVE-BLOCK-END
