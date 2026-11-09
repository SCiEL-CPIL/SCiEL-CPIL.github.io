"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator


class Optimizer:
    """
    Surrogate-assisted search combining:
    - Initial random sampling to build a model
    - RBF surrogate for cheap predictions
    - Local refinement using scipy minimize (Nelder-Mead) around promising points
    - Iterative refinement with budget management
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def _count_call(self, func, x):
        """Evaluate func(x) and track call count."""
        self.calls += 1
        return func(x)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.calls = 0

        # Phase 1: Initial random sampling to build initial model
        n_init = min(50, budget)
        X = np.random.default_rng(42).uniform(-5, 5, size=(n_init, dim))
        F = np.array([self._count_call(func, x) for x in X])

        best_idx = np.argmin(F)
        x_best = X[best_idx].copy()
        f_best = F[best_idx]

        # Phase 2: Iterative surrogate-guided search
        # We'll use RBF interpolation as a simple surrogate
        # and propose new points by minimizing the surrogate + exploration term

        remaining = budget - self.calls
        n_iter = 0

        while remaining > 0 and n_iter < 200:
            # Build/update RBF surrogate
            if len(X) >= 3:
                try:
                    rbf = RBFInterpolator(X, F, kernel='thin_plate_spline')
                except Exception:
                    # Fallback to linear if RBF fails
                    try:
                        rbf = RBFInterpolator(X, F, kernel='linear')
                    except Exception:
                        rbf = None
            else:
                rbf = None

            # Propose candidate points
            candidates = []

            # 1. Local refinement around best point using Nelder-Mead
            if remaining >= 5:
                try:
                    # Use a local optimizer around the current best
                    def surrogate_local(x):
                        if rbf is not None:
                            return rbf(x.reshape(1, -1))[0]
                        else:
                            return f_best

                    res = minimize(
                        surrogate_local,
                        x_best,
                        method='Nelder-Mead',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 10, 'xatol': 1e-3, 'fatol': 1e-6}
                    )
                    x_candidate = np.clip(res.x, -5, 5)
                    candidates.append(('local', x_candidate))
                except Exception:
                    pass

            # 2. Random perturbation around best point
            for scale in [0.5, 1.0, 2.0]:
                x_pert = x_best + scale * np.random.default_rng(n_iter + 100).standard_normal(dim)
                x_pert = np.clip(x_pert, -5, 5)
                candidates.append(('perturb', x_pert))

            # 3. If we have a surrogate, try to find a good exploration point
            if rbf is not None and remaining >= 3:
                try:
                    # Sample random points and pick the one with lowest predicted value
                    rng_explore = np.random.default_rng(n_iter + 200)
                    n_explore = min(20, remaining)
                    X_explore = rng_explore.uniform(-5, 5, size=(n_explore, dim))
                    F_explore = rbf(X_explore)
                    best_explore_idx = np.argmin(F_explore)
                    candidates.append(('surrogate', X_explore[best_explore_idx].copy()))
                except Exception:
                    pass

            # 4. Pure random exploration
            if remaining >= 1:
                x_rand = np.random.default_rng(n_iter + 300).uniform(-5, 5, dim)
                candidates.append(('random', x_rand))

            # Evaluate candidates (pick the most promising based on type priority)
            # Prioritize: local > surrogate > perturb > random
            candidate_order = {'local': 0, 'surrogate': 1, 'perturb': 2, 'random': 3}
            candidates.sort(key=lambda c: candidate_order.get(c[0], 99))

            improved = False
            for ctype, x_cand in candidates:
                if self.calls >= budget:
                    break

                f_cand = self._count_call(func, x_cand)

                # Add to model
                X = np.vstack([X, x_cand.reshape(1, -1)])
                F = np.append(F, f_cand)

                if f_cand < f_best:
                    f_best = f_cand
                    x_best = x_cand.copy()
                    improved = True
                    break  # Move to next iteration with the new best

            remaining = budget - self.calls
            n_iter += 1

        return x_best, f_best


# EVOLVE-BLOCK-END
