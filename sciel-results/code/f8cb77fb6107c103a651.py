"""Surrogate-assisted multi-objective Bayesian optimization for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Surrogate-assisted multi-objective optimizer using polynomial regression and EHVI-like acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = 0.5 * (lo + hi)
        half_range = 0.5 * (hi - lo)

        # Reference point for hypervolume
        ref = np.array([3.0, 5.0, 7.0])

        # Initialize with random points
        init_size = min(30, self.budget)
        X = np.array([lo + (hi - lo) * rng.random(self.dim) for _ in range(init_size)])
        F = np.array([problem(x) for x in X])

        used = init_size
        best_front = self._nondominated(F)

        # Surrogate model coefficients (polynomial degree 2 in normalized space)
        # We'll fit a simple linear/quadratic model for each objective

        while used < self.budget:
            if time.time() > deadline - 10:
                break

            # Fit surrogate models
            X_norm = (X - mid) / half_range  # Normalize to [-1, 1]
            surrogate = self._fit_surrogate(X_norm, F)

            # Generate candidate points using acquisition function
            n_candidates = 50
            candidates = []
            for _ in range(n_candidates):
                # Mix of random and local perturbations around best points
                if rng.random() < 0.5:
                    # Random exploration
                    cand = lo + (hi - lo) * rng.random(self.dim)
                else:
                    # Exploit: perturb a non-dominated point
                    nd_idx = np.where(np.isin(np.arange(len(F)), best_front))[0]
                    parent_idx = nd_idx[rng.integers(len(nd_idx))]
                    sigma = 0.1 * (1.0 - 0.5 * used / self.budget)  # Decreasing sigma
                    cand = X[parent_idx] + sigma * half_range * rng.standard_normal(self.dim)
                cand = np.clip(cand, lo, hi)
                candidates.append(cand)

            candidates = np.array(candidates)

            # Score candidates using EHVI-like acquisition
            scores = np.zeros(len(candidates))
            for i, cand in enumerate(candidates):
                cand_norm = (cand - mid) / half_range
                predicted_F = surrogate(cand_norm)
                # Penalize if predicted point is dominated by current front
                if self._is_dominated(predicted_F, F[best_front]):
                    scores[i] = -1.0
                else:
                    # Approximate hypervolume contribution
                    scores[i] = self._approx_hv_contribution(predicted_F, F[best_front], ref)

            # Select best candidate
            best_idx = np.argmax(scores)
            new_point = candidates[best_idx]
            new_F = problem(new_point)
            used += 1

            # Add to population
            X = np.vstack([X, new_point])
            F = np.vstack([F, new_F])

            # Update best front
            best_front = self._nondominated(F)

            # Periodically prune population to keep it manageable
            if len(X) > 100:
                # Keep non-dominated and some random dominated points
                nd_mask = np.zeros(len(X), dtype=bool)
                nd_mask[best_front] = True
                # Add some random dominated points for diversity
                dominated_idx = np.where(~nd_mask)[0]
                if len(dominated_idx) > 0:
                    n_keep = min(20, len(dominated_idx))
                    keep_idx = rng.choice(dominated_idx, size=n_keep, replace=False)
                    nd_mask[keep_idx] = True
                X = X[nd_mask]
                F = F[nd_mask]
                best_front = self._nondominated(F)

        return X

    def _fit_surrogate(self, X_norm, F):
        """Fit a quadratic polynomial surrogate for each objective."""
        n, d = X_norm.shape
        m = F.shape[1]

        # Build feature matrix: constant, linear, quadratic terms
        features = np.ones((n, 1))
        features = np.hstack([features, X_norm])
        # Add quadratic terms
        for i in range(d):
            features = np.hstack([features, X_norm[:, i:i+1] ** 2])
        for i in range(d):
            for j in range(i+1, d):
                features = np.hstack([features, X_norm[:, i] * X_norm[:, j]])

        # Fit each objective
        coeffs = np.zeros((m, features.shape[1]))
        for k in range(m):
            try:
                coeffs[k] = np.linalg.lstsq(features, F[:, k], rcond=None)[0]
            except:
                coeffs[k] = np.mean(F[:, k])

        def surrogate(x_norm):
            x_norm = np.atleast_2d(x_norm)
            feat = np.ones((x_norm.shape[0], 1))
            feat = np.hstack([feat, x_norm])
            for i in range(d):
                feat = np.hstack([feat, x_norm[:, i:i+1] ** 2])
            for i in range(d):
                for j in range(i+1, d):
                    feat = np.hstack([feat, x_norm[:, i] * x_norm[:, j]])
            return feat @ coeffs.T

        return surrogate

    def _nondominated(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)

        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _is_dominated(self, point, front):
        """Check if point is dominated by any point in front."""
        for f in front:
            if np.all(f <= point) and np.any(f < point):
                return True
        return False

    def _approx_hv_contribution(self, new_point, front, ref):
        """Approximate hypervolume contribution of new_point given current front."""
        # Simple approximation: compute HV of front + new_point minus HV of front
        # This is expensive, so use a simpler heuristic
        # For now, use negative sum of objectives as a proxy
        return -np.sum(new_point)
# EVOLVE-BLOCK-END
