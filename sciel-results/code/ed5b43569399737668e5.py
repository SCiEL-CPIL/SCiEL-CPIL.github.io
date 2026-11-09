"""Multi-start local search for WFG4 using L-BFGS-B on Tchebycheff scalarization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start local search: L-BFGS-B on Tchebycheff scalarization from LHS seeds."""

    def __init__(self, dim, n_obj):
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300.0 - 5.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        bounds_list = list(zip(lo, hi))

        # Collect all evaluated points
        all_X = []
        all_F = []

        def record(x, f):
            all_X.append(np.asarray(x, dtype=float))
            all_F.append(np.asarray(f, dtype=float))

        # --- Initial set: LHS (low-discrepancy) for basin coverage ---
        n_initial = 8
        rng = np.random.default_rng(42)
        # Latin hypercube sampling
        idx = [rng.permutation(n_initial) for _ in range(dim)]
        X_lhs = np.array([(i + 0.5) / n_initial for i in range(n_initial)])
        X_init = np.zeros((n_initial, dim))
        for d in range(dim):
            X_init[:, d] = lo[d] + (hi[d] - lo[d]) * X_lhs[idx[d]]

        for x in X_init:
            if time.time() > deadline:
                break
            f = problem(x)
            record(x, f)

        # --- Multi-start local search: Tchebycheff scalarization ---
        # Tchebycheff: min max_i w_i * (f_i - z_i)
        # We use several weight vectors to target different parts of the Pareto front

        # Weight vectors: simplex-lattice for 3 objectives
        weights = []
        for w1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for w2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                w3 = 1.0 - w1 - w2
                if w3 > 0.05:
                    weights.append([w1, w2, w3])
        # Add extreme weights
        weights.extend([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9],
                        [0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]])

        # For each (weight, initial point) pair, run L-BFGS-B
        eval_count = len(all_X)
        max_evals = 200
        max_fns_per_opt = 30  # L-BFGS-B function evals per optimization

        # Sort by priority: extreme weights first (likely better hypervolume corners)
        # Then mix with interior weights
        priority_order = [0, 1, 2, 3, 4, 5]  # extremes + balanced
        remaining = list(range(len(weights)))
        for pi in priority_order:
            if pi < len(remaining):
                remaining.remove(pi)
        ordered_weights = [weights[i] for i in priority_order if i < len(weights)]
        ordered_weights += [weights[i] for i in remaining]

        # Interleave: for each weight, try from a different initial point
        for w_idx, w in enumerate(ordered_weights):
            if eval_count >= max_evals:
                break
            if time.time() > deadline:
                break

            # Pick initial point (cycle through LHS points)
            x0 = X_init[w_idx % n_initial].copy()

            # Objective: Tchebycheff scalarization
            # z_i = 0 (we minimize, and WFG4 objectives are positive)
            def obj(x):
                nonlocal eval_count
                f = problem(x)
                eval_count += 1
                # Tchebycheff: max_i w_i * f_i
                return np.max(w * np.asarray(f))

            # We need to track evals carefully; L-BFGS-B calls obj multiple times
            # Use a counter
            fn_calls = [0]

            def obj_counted(x):
                fn_calls[0] += 1
                if fn_calls[0] > max_fns_per_opt:
                    raise RuntimeError("budget")
                f = problem(x)
                eval_count += 1
                if eval_count > max_evals:
                    raise RuntimeError("budget")
                return np.max(w * np.asarray(f))

            try:
                res = minimize(obj_counted, x0, method='L-BFGS-B',
                               bounds=bounds_list,
                               options={'maxiter': 50, 'maxfun': max_fns_per_opt})
                x_best = res.x
                f_best = problem(x_best)
                eval_count += 1
                record(x_best, f_best)
            except RuntimeError:
                # Budget exceeded inside; record last known good if we have one
                pass
            except Exception:
                pass

        # --- If we have budget left, do a second round of refinement ---
        if eval_count < max_evals and time.time() < deadline:
            # Refine the best points found so far with smaller steps
            if all_X:
                F_arr = np.array(all_F)
                X_arr = np.array(all_X)
                # Pick non-dominated points for refinement
                # Simple non-dominance check
                n_pts = len(F_arr)
                dominated = np.zeros(n_pts, dtype=bool)
                for i in range(n_pts):
                    for j in range(n_pts):
                        if i != j:
                            if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                                dominated[i] = True
                                break
                nd_idx = np.where(~dominated)[0]
                if len(nd_idx) == 0:
                    nd_idx = [np.argmin(np.sum(F_arr, axis=1))]

                for idx in nd_idx[:4]:  # refine up to 4 non-dominated points
                    if eval_count >= max_evals or time.time() > deadline:
                        break
                    x0 = X_arr[idx].copy()
                    w = [0.33, 0.33, 0.34]  # balanced refinement

                    fn_calls2 = [0]

                    def obj_refine(x):
                        fn_calls2[0] += 1
                        if fn_calls2[0] > 15:
                            raise RuntimeError("budget")
                        f = problem(x)
                        eval_count += 1
                        if eval_count > max_evals:
                            raise RuntimeError("budget")
                        return np.max(w * np.asarray(f))

                    try:
                        res = minimize(obj_refine, x0, method='L-BFGS-B',
                                       bounds=bounds_list,
                                       options={'maxiter': 30, 'maxfun': 15})
                        x_best = res.x
                        f_best = problem(x_best)
                        eval_count += 1
                        record(x_best, f_best)
                    except RuntimeError:
                        pass
                    except Exception:
                        pass

        return all_X
# EVOLVE-BLOCK-END
