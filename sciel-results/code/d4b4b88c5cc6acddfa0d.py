"""Optimizer: adaptive grid-search warm-up + CMA-ES refinement for BBOB f1 (Sphere)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Two-phase optimizer: adaptive grid search for localization, CMA-ES for refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Adaptive multi-resolution grid search ---
        # Start with a coarse grid, then refine around the best point
        
        # Coarse grid: evaluate at regular intervals
        # For dim=5, use a grid with ~3-4 points per dimension
        coarse_step = 2.5  # Points at -5, -2.5, 0, 2.5, 5
        coarse_points = []
        for d in range(dim):
            for v in [-5.0, -2.5, 0.0, 2.5, 5.0]:
                pt = np.zeros(dim)
                pt[d] = v
                coarse_points.append(pt)
        
        # Add center and some diagonal points
        coarse_points.append(np.zeros(dim))
        
        # Evaluate coarse grid
        for pt in coarse_points:
            if calls_made >= budget:
                break
            evaluate(pt)

        if best_x is None:
            best_x = np.zeros(dim)
            evaluate(best_x)

        # Refine around the best point with a finer grid
        # Use a smaller step size around best_x
        refine_center = best_x.copy()
        refine_step = 1.0
        refine_points = []
        
        # Generate points in a box around best_x
        for d in range(dim):
            for v in [-refine_step, -0.5 * refine_step, 0.0, 0.5 * refine_step, refine_step]:
                pt = refine_center.copy()
                pt[d] += v
                if np.all(pt >= -5.0) and np.all(pt <= 5.0):
                    refine_points.append(pt)
        
        # Also add some points at intermediate distances
        for d in range(dim):
            for sign in [-1, 1]:
                for dist in [1.5, 2.0]:
                    pt = refine_center.copy()
                    pt[d] += sign * dist
                    if np.all(pt >= -5.0) and np.all(pt <= 5.0):
                        refine_points.append(pt)

        # Evaluate refinement grid
        for pt in refine_points:
            if calls_made >= budget:
                break
            evaluate(pt)

        if best_x is None:
            best_x = np.zeros(dim)
            evaluate(best_x)

        # --- Phase 2: CMA-ES refinement ---
        # Estimate initial sigma from the spread of good points
        # Use the distance between best point and second-best as a guide
        sigma = 0.5  # Conservative initial step size after grid search

        x = best_x.copy()
        fx = best_f

        # CMA-ES parameters tuned for dim=5
        lam = 10  # Population size
        mu = lam // 2  # Number of parents

        # Logarithmic weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Learning rates (standard CMA-ES)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))

        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial mean and covariance
        mean = x.copy()
        C = np.eye(dim)

        # Chi-norm for sigma update
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        remaining = budget - calls_made
        if remaining <= 0:
            return x, fx

        # Run CMA-ES
        gen = 0
        while calls_made < budget:
            # Eigen decomposition of C
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-20)
            B = evecs
            D = np.sqrt(evals)

            # Sample offspring
            z = rng.standard_normal((lam, dim))
            y = z @ (B * D).T

            # Update sigma path
            y_sigma = y / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                      (weights[:mu] @ y_sigma[:mu].T)

            # Hessian test for sigma update
            norm_p_sigma = np.linalg.norm(p_sigma)
            expected_norm = chi_n * np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
            h_sigma = float(norm_p_sigma / expected_norm < (1.4 + 2.0 / (dim + 1)) * np.sqrt(mu_eff))

            # Update C path
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                  (weights[:mu] @ y[:mu].T)

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_c = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])

            C = (1 + c_1 * (c_mu - c_1) - c_1) * C + c_1 * (rank_c + delta_h * rank_c) + \
                c_mu * (rank_mu - C)

            # Ensure symmetry and PSD
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)

            # Update sigma
            sigma *= np.exp((c_sigma / 5.0) * (norm_p_sigma / chi_n - 1.0))

            # Generate and evaluate candidates
            candidates = []
            fvals = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                xi = mean + sigma * y[i]
                xi = np.clip(xi, -5.0, 5.0)
                fi = func(xi)
                calls_made += 1
                candidates.append(xi)
                fvals.append(fi)

            if len(fvals) == 0:
                break

            # Sort offspring by fitness
            idx = np.argsort(fvals)
            y_sorted = y[idx]
            fvals_sorted = np.array(fvals)[idx]

            # Update mean
            mean = mean + sigma * (weights[:mu] @ y_sorted[:mu].T)
            mean = np.clip(mean, -5.0, 5.0)

            # Update best
            if fvals_sorted[0] < fx:
                fx = fvals_sorted[0]
                x = candidates[idx[0]].copy()

            # Stopping criteria
            if sigma < 1e-12:
                break
            if gen > 10 and np.max(np.abs(y_sorted[:mu])) < 1e-10:
                break
                
            gen += 1

        return x, fx
# EVOLVE-BLOCK-END
