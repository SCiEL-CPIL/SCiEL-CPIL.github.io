"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol + corner initialization with efficient DE-based multi-objective search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Initialize with Sobol + corners for diverse coverage
        n_corners = min(2 ** dim, 32)
        corners = np.array(np.meshgrid(*[np.array([lo[i], hi[i]]) for i in range(dim)])).T.reshape(-1, dim)
        if n_corners < len(corners):
            idx = rng.choice(len(corners), n_corners, replace=False)
            corners = corners[idx]

        pop_size = min(64, budget // 3)
        n_sobol = max(1, pop_size - n_corners)
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sampler.random(n_sobol)
        sobol_pts = lo + (hi - lo) * sobol_samples

        X = np.vstack([corners, sobol_pts])
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Efficient non-dominated set maintenance
        def is_dominated(f_new, F_existing):
            """Check if f_new is dominated by any point in F_existing."""
            # Vectorized check: is there any point in F_existing that dominates f_new?
            diff = F_existing - f_new
            return np.any(np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1))

        def dominates(f_new, F_existing):
            """Check which points in F_existing are dominated by f_new."""
            diff = F_existing - f_new
            return np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)

        # Main DE loop
        CR = 0.8
        F_mut = 0.8

        while used < budget:
            if time.time() > deadline:
                break

            # Get non-dominated indices efficiently
            n = len(F)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                diff = F - F[i]
                dom = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                dom[i] = False
                if np.any(dom):
                    nd_mask[i] = False

            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) < 3:
                parent_pool = np.arange(n)
            else:
                parent_pool = nd_indices

            n_parents = len(parent_pool)
            if n_parents < 3:
                break

            # DE mutation in decision space
            idx1, idx2, idx3 = rng.choice(n_parents, size=3, replace=False)
            x1 = X[parent_pool[idx1]]
            x2 = X[parent_pool[idx2]]
            x3 = X[parent_pool[idx3]]
            v = x1 + F_mut * (x2 - x3)

            # Binomial crossover
            trial = X[parent_pool[idx1]].copy()
            crossover_mask = rng.random(dim) < CR
            if not np.any(crossover_mask):
                crossover_mask[rng.integers(dim)] = True
            trial[crossover_mask] = v[crossover_mask]
            trial = np.clip(trial, lo, hi)

            ft = problem(trial)
            used += 1

            # Update population: replace worst point
            # Check if trial is dominated by existing
            if is_dominated(ft, F):
                # Trial is dominated, replace the most dominated point
                dom_counts = np.zeros(n)
                for i in range(n):
                    if nd_mask[i]:
                        continue
                    diff = F - F[i]
                    dom_counts[i] = np.sum(np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1))
                worst_idx = np.argmax(dom_counts)
                if dom_counts[worst_idx] > 0:
                    X[worst_idx] = trial
                    F[worst_idx] = ft
            else:
                # Trial is non-dominated, remove points it dominates and add it
                dominated_by_trial = dominates(ft, F)
                if np.any(dominated_by_trial):
                    # Remove one dominated point (the one with worst spread)
                    dominated_indices = np.where(dominated_by_trial)[0]
                    # Replace the most dominated one
                    dom_counts = np.zeros(n)
                    for i in dominated_indices:
                        diff = F - F[i]
                        dom_counts[i] = np.sum(np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1))
                    worst_idx = dominated_indices[np.argmax(dom_counts[dominated_indices])]
                    X[worst_idx] = trial
                    F[worst_idx] = ft
                else:
                    # Replace a random non-dominated point to maintain diversity
                    nd_existing = np.where(nd_mask)[0]
                    if len(nd_existing) > 0:
                        replace_idx = nd_existing[rng.integers(len(nd_existing))]
                        X[replace_idx] = trial
                        F[replace_idx] = ft

        return X

# EVOLVE-BLOCK-END
