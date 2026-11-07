"""Decomposition-based multi-objective optimizer using Tchebycheff scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Tchebycheff decomposition with Nelder-Mead local search per sub-problem."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = 0.5 * (lo + hi)

        # Generate weight vectors uniformly on the simplex
        n_weights = 30
        # Use Dirichlet sampling with alpha=1 for uniform distribution
        rng = np.random.default_rng(42)
        raw = rng.dirichlet(np.ones(n_obj), size=n_weights)
        weights = raw  # shape (n_weights, n_obj)

        # Evaluate the center point first
        f_center = np.array(problem(center))

        # For each weight vector, run Nelder-Mead on the Tchebycheff scalarized objective
        # Budget allocation: reserve some for initial evaluations, rest for local search
        # Each local search uses a limited number of function evaluations

        # We have budget total evaluations. Let's allocate:
        # - 1 for center
        # - Remaining budget / n_weights for each sub-problem
        remaining = budget - 1
        evals_per_sub = max(1, remaining // n_weights)

        all_x = [center]
        all_f = [f_center]

        for w in weights:
            # Tchebycheff scalarization: min max_i w_i * (f_i(x) - z_i*)
            # We need a reference point z*. Use the best known or a fixed reference.
            # Since we don't know the ideal point, we'll use a dynamic reference.
            # For simplicity, use the center's objective values as initial reference.

            def tcheb_obj(x):
                fx = np.array(problem(x))
                # Use f_center as reference for now
                return np.max(w * (fx - f_center))

            # Initial point: center
            x0 = center.copy()

            # Nelder-Mead with bounds via clipping
            # scipy's minimize doesn't handle bounds well with Nelder-Mead, so we clip
            # We'll wrap the objective to clip x to bounds
            def obj_wrapped(x):
                xc = np.clip(x, lo, hi)
                return tcheb_obj(xc)

            try:
                res = minimize(
                    obj_wrapped,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': evals_per_sub,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                x_best = np.clip(res.x, lo, hi)
                f_best = np.array(problem(x_best))
                all_x.append(x_best)
                all_f.append(f_best)
            except Exception:
                # If local search fails, just keep the center
                pass

        return all_x
# EVOLVE-BLOCK-END
