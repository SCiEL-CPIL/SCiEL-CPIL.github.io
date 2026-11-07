"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer for DTLZ2 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _crowding(self, F):
        """Crowding distance for a set of points (assumed non-dominated)."""
        m = F.shape[0]
        if m <= 2:
            return np.full(m, np.inf)
        d = np.zeros(m)
        for k in range(F.shape[1]):
            order = np.argsort(F[:, k])
            d[order[0]] = np.inf
            d[order[-1]] = np.inf
            rng_span = F[order[-1], k] - F[order[0], k]
            if rng_span > 0:
                for i in range(1, m - 1):
                    d[order[i]] += (F[order[i + 1], k] - F[order[i - 1], k]) / rng_span
        return d

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin under 300s per eval / 330s total

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim, n_obj, budget = self.dim, self.n_obj, self.budget
        rng = np.random.default_rng(0)

        # Population size: large enough to cover the front, small enough to fit budget
        pop = int(min(80, max(20, budget // 3)))
        X = lo + (hi - lo) * rng.random((pop, dim))
        F = np.array([problem(x) for x in X])
        used = pop

        # Adaptive mutation scale
        scale = 0.7

        while used < budget:
            if time.time() > deadline:
                break

            # Identify non-dominated set for selection
            nd_mask = np.ones(pop, dtype=bool)
            for i in range(pop):
                if not nd_mask[i]:
                    continue
                # does any j dominate i?
                dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                dom[i] = False
                if np.any(dom):
                    nd_mask[i] = False

            if np.any(nd_mask):
                parents = X[nd_mask]
            else:
                parents = X

            # Generate children via DE
            n_children = min(10, budget - used)
            children = np.empty((n_children, dim))
            for c in range(n_children):
                # DE/rand/1
                idxs = rng.integers(0, len(parents), size=3)
                a, b, cc = parents[idxs[0]], parents[idxs[1]], parents[idxs[2]]
                v = a + scale * (b - cc)
                # crossover
                trial = np.where(rng.random(dim) < 0.5, v, parents[c % len(parents)])
                children[c] = np.clip(trial, lo, hi)

            Fc = np.array([problem(x) for x in children])
            used += n_children

            # Combine and select
            Xc = np.vstack([X, children])
            Fc_all = np.vstack([F, Fc])

            # Non-dominated sort
            nd_mask = np.ones(len(Xc), dtype=bool)
            for i in range(len(Xc)):
                if not nd_mask[i]:
                    continue
                dom = np.all(Fc_all <= Fc_all[i], axis=1) & np.any(Fc_all < Fc_all[i], axis=1)
                dom[i] = False
                if np.any(dom):
                    nd_mask[i] = False

            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) <= pop:
                X, F = Xc[nd_idx], Fc_all[nd_idx]
                # pad if needed (shouldn't be)
            else:
                # keep top by crowding
                nd_F = Fc_all[nd_idx]
                cd = self._crowding(nd_F)
                keep = np.argsort(cd)[::-1][:pop]
                X = Xc[nd_idx[keep]]
                F = Fc_all[nd_idx[keep]]

            # Occasionally reduce scale if stagnating (simple heuristic)
            if used % 50 == 0:
                scale = max(0.4, scale * 0.95)

        # Final: return best non-dominated set
        if len(F) > 0:
            nd_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not nd_mask[i]:
                    continue
                dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                dom[i] = False
                if np.any(dom):
                    nd_mask[i] = False
            if np.any(nd_mask):
                X = X[nd_mask]
                F = F[nd_mask]

        return X
# EVOLVE-BLOCK-END
