"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with deadline guard and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Initial population: Latin Hypercube sampling for better coverage
        pop_size = min(20, self.budget // 5)
        X = np.empty((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[perm, d] = lo[d] + span[d] * (np.arange(pop_size) + 0.5) / pop_size
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated points with vectorized check
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                diff = F[i] - F
                mask = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                mask[i] = False
                dominated[mask] = True

            non_dom_idx = np.where(~dominated)[0]
            n_nd = len(non_dom_idx)
            
            # Select parent: tournament among non-dominated
            if n_nd > 0:
                if n_nd <= 3:
                    parent_idx = non_dom_idx[rng.integers(n_nd)]
                else:
                    candidates = non_dom_idx[rng.integers(n_nd, size=3)]
                    sums = np.sum(F[candidates], axis=1)
                    parent_idx = candidates[np.argmin(sums)]
            else:
                parent_idx = rng.integers(n)

            # Adaptive mutation with local spread awareness
            progress = used / self.budget
            base_step = 0.12 * (1.0 - 0.6 * progress)
            
            if n_nd >= 2:
                nd_points = F[non_dom_idx]
                local_spread = np.std(nd_points, axis=0)
                local_range = np.ptp(nd_points, axis=0)
                local_range[local_range < 1e-10] = 1e-10
                adaptive_factor = np.clip(np.mean(local_spread / local_range), 0.3, 1.0)
            else:
                adaptive_factor = 1.0

            r = rng.random()
            if r < 0.15 and n_nd >= 2:
                other_idx = non_dom_idx[rng.integers(n_nd)]
                alpha = rng.random()
                child = alpha * X[parent_idx] + (1 - alpha) * X[other_idx]
                child += 0.03 * span * rng.standard_normal(dim)
            elif r < 0.55:
                step = base_step * adaptive_factor
                child = X[parent_idx] + step * span * rng.standard_normal(dim)
            elif r < 0.8:
                child = X[parent_idx] + 0.2 * span * rng.standard_normal(dim)
            else:
                child = X[parent_idx] + 0.4 * span * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point
            worst_score = np.sum(F, axis=1)
            for i in range(n):
                if dominated[i]:
                    worst_score[i] += 1e10
            worst_idx = int(np.argmax(worst_score))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
