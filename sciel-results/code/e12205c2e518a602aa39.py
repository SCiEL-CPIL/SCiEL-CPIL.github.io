"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using LHS init, adaptive mutation, and HV-guided selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _lhs(n, dim):
        """Latin Hypercube Sampling."""
        pts = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            pts[:, d] = (perm + np.random.random(n)) / n
        return pts

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit with safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng()

        # Population size
        pop_size = min(40, max(10, self.budget // 5))
        remaining = self.budget

        # Initialize with LHS
        X = lo + (hi - lo) * self._lhs(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        remaining -= pop_size

        # Adaptive mutation step size
        step = 0.2 * (hi - lo)

        # Track best hypervolume contribution for selection
        def compute_hv_contrib(F_arr, ref):
            """Compute hypervolume contribution of each point (approximate)."""
            n = len(F_arr)
            contribs = np.zeros(n)
            # Sort by first objective
            order = np.argsort(F_arr[:, 0])
            for i in range(n):
                f = F_arr[order[i]]
                if np.any(f >= ref):
                    continue
                # Volume dominated by this point but not by any previously considered (with better first obj)
                # Simplified: box volume from f to ref
                vol = np.prod(ref - f)
                contribs[order[i]] = vol
            return contribs

        while remaining > 0:
            # Check time
            if time.time() > deadline - 5:
                break

            # Find non-dominated set
            nd_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not nd_mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd_mask[i] = False
                        break

            # Select parent from non-dominated set using HV-guided approach
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.arange(len(F))

            # Weight selection by objective diversity
            # Use tournament: pick 2 random from ND, keep better in terms of sum
            if len(nd_indices) > 1:
                a = nd_indices[rng.integers(len(nd_indices))]
                b = nd_indices[rng.integers(len(nd_indices))]
                if np.sum(F[a]) <= np.sum(F[b]):
                    parent = X[a]
                else:
                    parent = X[b]
            else:
                parent = X[nd_indices[0]]

            # Adaptive mutation: occasionally restart from random
            if rng.random() < 0.05:
                child = lo + (hi - lo) * rng.random(self.dim)
            else:
                # Gaussian mutation with adaptive step
                child = parent + step * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

            fc = problem(child)
            remaining -= 1

            # Replace worst point (by HV contribution proxy: max sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Adaptive step: shrink if we keep improving, expand if stagnating
            # Simple adaptation based on recent performance
            if np.sum(fc) < np.sum(F[worst_idx]):  # improvement
                step *= 0.95
            else:
                step *= 1.05
            step = np.clip(step, 0.01 * (hi - lo), 0.5 * (hi - lo))

        return X
# EVOLVE-BLOCK-END
