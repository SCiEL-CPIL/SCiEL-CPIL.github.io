"""Multi-objective optimizer on RE21: explore-then-exploit with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase MO optimizer: LHS exploration + adaptive local search on the front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin vs 300s / 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)

        used = 0
        X = []
        F = []

        def eval_point(x):
            nonlocal used
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        def dominated_mask(Fmat):
            """Return boolean mask: True if point i is dominated by some other point."""
            n = len(Fmat)
            dom = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(Fmat[j] <= Fmat[i]) and np.any(Fmat[j] < Fmat[i]):
                        dom[i] = True
                        break
            return dom

        def non_dominated(Fmat):
            return ~dominated_mask(Fmat)

        # Phase 1: diverse exploration via stratified (LHS-like) sampling
        n_explore = min(80, self.budget)
        n = n_explore
        # stratified random: split each dim into n cells, pick one random point per cell
        for k in range(n):
            if time.time() > deadline:
                break
            # one random point per dimension cell
            x = np.empty(self.dim)
            for d in range(self.dim):
                cell_lo = lo[d] + (hi[d] - lo[d]) * (k / n)
                cell_hi = lo[d] + (hi[d] - lo[d]) * ((k + 1) / n)
                x[d] = rng.uniform(cell_lo, cell_hi)
            f = eval_point(x)
            X.append(x)
            F.append(f)

        X = np.array(X)
        F = np.array(F)

        # Phase 2: local search exploiting non-dominated front
        step = 0.15
        best_step = step
        no_improve = 0

        while used < self.budget:
            if time.time() > deadline:
                break

            Fmat = np.array(F)
            nd_idx = np.where(non_dominated(Fmat))[0]

            # choose a parent: prefer non-dominated; random among them
            if len(nd_idx) > 0:
                parent_i = int(rng.choice(nd_idx))
            else:
                parent_i = int(rng.integers(len(X)))

            parent = X[parent_i]

            # adaptive mutation: try a few step sizes, keep first acceptable child
            child = None
            for trial in range(3):
                s = step * (1.0 if trial == 0 else 0.5 ** trial)
                cand = parent + s * (hi - lo) * rng.standard_normal(self.dim)
                cand = np.clip(cand, lo, hi)
                fc = eval_point(cand)
                used_check = used
                # accept if child is not dominated by current set
                Fmat_now = np.array(F)
                # check dominance of child by existing
                child_dominated = np.any(
                    (np.all(Fmat_now <= fc, axis=1) & np.any(Fmat_now < fc, axis=1))
                )
                if not child_dominated:
                    child = cand
                    fc_child = fc
                    break
                else:
                    # still record child as candidate but we'll decide replacement
                    child = cand
                    fc_child = fc

            # Decide replacement:
            # If child non-dominated: replace the dominated point with worst sum, or append if room
            Fmat_now = np.array(F)
            child_dominated_now = np.any(
                (np.all(Fmat_now <= fc_child, axis=1) & np.any(Fmat_now < fc_child, axis=1))
            )

            if not child_dominated_now:
                # child is non-dominated; replace a dominated point if any, else the worst
                dom_mask = dominated_mask(Fmat_now)
                if np.any(dom_mask):
                    # among dominated, pick the one with worst (largest) sum
                    dom_indices = np.where(dom_mask)[0]
                    sums = np.sum(Fmat_now[dom_indices], axis=1)
                    worst_local = int(np.argmax(sums))
                    replace_i = int(dom_indices[worst_local])
                else:
                    # all non-dominated; replace the one furthest in sum (least extreme)
                    sums = np.sum(Fmat_now, axis=1)
                    replace_i = int(np.argmax(sums))
                X[replace_i] = child
                F[replace_i] = fc_child
                no_improve = 0
                best_step = step
                # slightly grow step to explore more
                step = min(0.3, step * 1.05)
            else:
                no_improve += 1
                step = max(0.02, step * 0.85)

        return X
# EVOLVE-BLOCK-END
