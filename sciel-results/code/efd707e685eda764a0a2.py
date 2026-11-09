"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using LHS-based multi-start local search on WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Latin Hypercube Sampling for initial diverse points
        n_lhs = min(60, budget // 4)
        rng = np.random.default_rng(42)

        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(n_lhs):
            for j in range(dim):
                lhs_points[i, j] = lo[j] + (hi[j] - lo[j]) * (i + rng.random()) / n_lhs
        # Shuffle to decorrelate
        rng.shuffle(lhs_points)

        # Evaluate initial points
        archive = []
        used = 0

        def add_to_archive(x, f):
            nonlocal used
            # Check if f is better than reference point
            ref = np.array([3.0, 5.0, 7.0])
            if np.any(f >= ref):
                return
            # Check dominance against archive
            dominated = False
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            if not dominated:
                # Remove archive points dominated by f
                archive[:] = [af for af in archive if not (np.all(f <= af) and np.any(f < af))]
                archive.append(f)

        for x in lhs_points:
            if used >= budget:
                break
            if time.time() > deadline - 30:
                break
            f = problem(x)
            used += 1
            add_to_archive(x, f)

        # Multi-start local search: use scipy minimize on each objective
        # We'll do a weighted sum approach with different weights
        # Focus on finding points along the Pareto front

        # Generate weight vectors for scalarization
        n_weights = min(15, (budget - used) // 10)
        if n_weights > 0:
            # Create weight vectors that emphasize different parts of the front
            weights = []
            for i in range(n_weights):
                w = np.ones(n_obj) / n_obj
                # Bias toward different objectives
                bias_idx = i % n_obj
                w[bias_idx] *= 3.0
                w /= np.sum(w)
                weights.append(w)

            # For each weight, do a few local searches from different starting points
            starts_per_weight = max(1, (budget - used) // (n_weights * 5))

            for w in weights:
                if used >= budget:
                    break
                if time.time() > deadline - 10:
                    break

                # Pick starting points from archive or random
                for _ in range(starts_per_weight):
                    if used >= budget:
                        break
                    if time.time() > deadline - 10:
                        break

                    # Random start
                    x0 = lo + (hi - lo) * rng.random(dim)

                    def scalar_obj(x, w=w):
                        fx = problem(x)
                        return float(np.dot(w, fx))

                    try:
                        res = minimize(scalar_obj, x0, method='Nelder-Mead',
                                       bounds=list(bounds),
                                       options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                        if res.success or res.fun < np.inf:
                            x_best = np.clip(res.x, lo, hi)
                            f_best = problem(x_best)
                            used += 1
                            add_to_archive(x_best, f_best)
                    except Exception:
                        pass

        # Additional random perturbations around archive points if budget remains
        if used < budget and len(archive) > 0:
            # Convert archive back to x points is not straightforward, 
            # so do random exploration in promising regions
            n_extra = min(budget - used, 20)
            for _ in range(n_extra):
                if time.time() > deadline - 5:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                f = problem(x)
                used += 1
                add_to_archive(x, f)

        return archive
# EVOLVE-BLOCK-END
