# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II-style multi-objective search with vectorized dominance,
    crowding-distance selection, adaptive mutation, and a wall-clock guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _non_dominated_mask(F):
        """Return boolean mask of non-dominated rows in F (n x m, minimize)."""
        n = F.shape[0]
        # A[i, j] is True if i dominates j: F[i] <= F[j] elementwise AND F[i] < F[j] somewhere
        le = F[:, None, :] <= F[None, :, :]          # n x n x m
        lt = F[:, None, :] < F[None, :, :]
        dom = np.all(le, axis=2) & np.any(lt, axis=2)  # n x n
        # A row i is dominated if any other row j dominates it
        dominated = np.any(dom, axis=0)
        return ~dominated

    @staticmethod
    def _crowding(F, mask):
        """Crowding distance for rows where mask is True; 0 elsewhere."""
        n = F.shape[0]
        m = F.shape[1]
        dist = np.zeros(n)
        if not np.any(mask):
            return dist
        idx = np.where(mask)[0]
        k = len(idx)
        if k <= 2:
            dist[idx] = np.inf
            return dist
        for j in range(m):
            col = F[idx, j]
            order = np.argsort(col)
            mn, mx = col[order[0]], col[order[-1]]
            if mx > mn:
                # neighbors in sorted order
                for t in range(1, k - 1):
                    d = (col[order[t + 1]] - col[order[t - 1]]) / (mx - mn)
                    dist[idx[order[t]]] += d
                dist[idx[order[0]]] = np.inf
                dist[idx[order[-1]]] = np.inf
        return dist

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0   # safety margin under 300s per evaluation / 330s kill
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        # Initial random batch
        pop = int(min(50, max(10, self.budget // 4)))
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop

        # Adaptive mutation sigma: start wide, shrink over time
        base_sigma = 0.25
        # We'll re-seed mutation strength occasionally to escape plateaus
        restart_every = 40
        last_restart = used
        sigma = base_sigma

        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated front among current population
            nd_mask = self._non_dominated_mask(F)
            if np.any(nd_mask):
                parents_idx = np.where(nd_mask)[0]
            else:
                parents_idx = np.arange(len(F))

            # Crowding distance to pick diverse parent
            cd = self._crowding(F, nd_mask)
            # Weighted random pick: prefer high crowding, but keep some randomness
            weights = cd[parents_idx] + 1e-9
            parent_idx = parents_idx[rng.choice(len(parents_idx), p=weights / weights.sum())]
            parent = X[parent_idx]

            # Mutate
            child = parent + sigma * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replacement: remove a dominated individual with low crowding distance
            # Prefer to remove dominated points; if all are non-dominated, remove the least crowded
            dom_mask = ~nd_mask
            if np.any(dom_mask):
                # Among dominated, pick one at random (or least crowded)
                cand = np.where(dom_mask)[0]
                # Pick the one with lowest crowding distance
                cd_all = self._crowding(F, nd_mask)
                # For dominated points, assign 0 crowding so we just pick min-sum as tiebreak
                replace_idx = cand[np.argmin(np.sum(F[cand], axis=1))]
            else:
                # All non-dominated: remove the one with smallest crowding distance
                cd_all = self._crowding(F, nd_mask)
                replace_idx = int(np.argmin(cd_all))

            X[replace_idx] = child
            F[replace_idx] = fc

            # Adaptive sigma: shrink slightly each step; restart every restart_every
            sigma *= 0.995
            if used - last_restart >= restart_every and used < self.budget:
                # Occasionally boost sigma to escape local plateaus
                sigma = base_sigma * rng.uniform(0.5, 1.0)
                last_restart = used

        return X
# EVOLVE-BLOCK-END
