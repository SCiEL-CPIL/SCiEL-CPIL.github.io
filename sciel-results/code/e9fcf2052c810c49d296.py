# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for WFG4 using constrained Chebyshev scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 260  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        def chebyshev_objective(x, weights, anchor):
            """Constrained Chebyshev scalarization objective."""
            f = np.array(problem(x), dtype=float)
            # Chebyshev: max_i (weights_i * (f_i - anchor_i))
            scaled = weights * (f - anchor)
            return np.max(scaled)

        # Generate weight vectors on the simplex
        # Use systematic sampling: Dirichlet-like spread
        n_weights = 12
        weights_list = []
        
        # Create weight vectors that cover the simplex
        # For 3 objectives, weights are (w1, w2, w3) with w1+w2+w3=1, wi>=0
        # Use a grid-like approach on the 2D simplex
        n_per_edge = 4
        for i in range(n_per_edge):
            for j in range(n_per_edge - i):
                k = n_per_edge - i - j
                w = np.array([i, j, k], dtype=float)
                w = w / w.sum()
                # Add small perturbation to avoid exact duplicates
                w = w + np.random.normal(0, 0.01, 3)
                w = np.clip(w, 0.01, None)
                w = w / w.sum()
                weights_list.append(w)
        
        # Deduplicate and select top n_weights
        seen = set()
        unique_weights = []
        for w in weights_list:
            key = tuple(np.round(w, 2))
            if key not in seen:
                seen.add(key)
                unique_weights.append(w)
        
        # If we have more than needed, subsample
        if len(unique_weights) > n_weights:
            idxs = np.random.choice(len(unique_weights), n_weights, replace=False)
            weights_list = [unique_weights[i] for i in idxs]
        else:
            weights_list = unique_weights

        # Estimate anchor point: use a rough estimate of the ideal point
        # Sample a few points to get a sense of the objective ranges
        n_probe = min(10, budget // 20)
        probe_X = lo + span * np.random.random((n_probe, dim))
        probe_F = np.array([problem(x) for x in probe_X], dtype=float)
        used += n_probe
        for i in range(n_probe):
            all_X.append(probe_X[i].copy())
            all_F.append(probe_F[i].copy())
        
        # Anchor: use the minimum of each objective from probes (approximate ideal point)
        anchor = np.min(probe_F, axis=0)
        # Shift anchor slightly to ensure we're working in a reasonable region
        anchor = np.maximum(anchor, 0.1)

        # Budget per weight vector
        remaining_budget = budget - used
        evals_per_weight = max(5, remaining_budget // len(weights_list))

        # For each weight vector, solve the scalar subproblem
        for w in weights_list:
            if used >= budget or time.time() > deadline:
                break

            # Multiple starting points for each subproblem
            n_starts = 2
            for s in range(n_starts):
                if used >= budget or time.time() > deadline:
                    break

                # Starting point: random or near previous best
                if s == 0:
                    x0 = lo + span * np.random.random(dim)
                else:
                    # Perturb a random point
                    x0 = lo + span * np.random.random(dim)
                    x0 += np.random.normal(0, 0.1, dim) * span
                    x0 = np.clip(x0, lo, hi)

                # Use Nelder-Mead for the scalar subproblem
                # We need to track evaluations carefully
                local_used = [0]
                local_x = [x0.copy()]
                local_f = [np.array(problem(x0), dtype=float)]
                local_used[0] = 1
                used += 1
                all_X.append(x0.copy())
                all_F.append(local_f[0].copy())

                def objective(x):
                    nonlocal used
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x), dtype=float)
                    used += 1
                    local_used[0] += 1
                    all_X.append(x.copy())
                    all_F.append(f.copy())
                    return chebyshev_objective(x, w, anchor)

                # Run Nelder-Mead with limited function calls
                max_iter = 30
                try:
                    res = minimize(objective, x0, method='Nelder-Mead',
                                   options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
                except Exception:
                    pass

        # Final cleanup: return all evaluated points
        return all_X
# EVOLVE-BLOCK-END
