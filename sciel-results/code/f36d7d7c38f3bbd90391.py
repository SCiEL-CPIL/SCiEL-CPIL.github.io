"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with adaptive mutation and a time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin vs 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        def dominated_mask(F):
            """Return boolean array: True where row i is dominated by some other row."""
            n = len(F)
            if n <= 1:
                return np.zeros(n, dtype=bool)
            le = F[:, None, :] <= F[None, :, :]   # (n,n,n): F[a] <= F[b]
            lt = F[:, None, :] < F[None, :, :]    # (n,n,n): F[a] <  F[b]
            dom = np.any(le & np.any(lt, axis=2), axis=1)  # b dominates a
            dom[np.arange(n), np.arange(n)] = False          # ignore self
            return dom

        # --- initial population ---
        pop_size = 40
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            if time.time() > deadline - 1.0:
                break

            # --- non-dominated sort ---
            remaining = list(range(len(F)))
            fronts = []
            while remaining:
                mask = dominated_mask(F[remaining])
                front = [remaining[i] for i in range(len(remaining)) if not mask[i]]
                fronts.append(front)
                remaining = [r for r in remaining if r not in set(front)]

            # --- crowding distance on the last (worst) front ---
            n = len(F)
            crowd = np.zeros(n)
            last = fronts[-1]
            if len(last) > 2:
                for k in range(self.n_obj):
                    order = np.argsort(F[last, k], kind='mergesort')
                    crowd[last[order[0]]] = np.inf
                    crowd[last[order[-1]]] = np.inf
                    obj_range = F[last[order[-1]], k] - F[last[order[0]], k]
                    if obj_range > 1e-12:
                        for j in range(1, len(order) - 1):
                            crowd[last[order[j]]] += (
                                F[last[order[j + 1]], k] - F[last[order[j - 1]], k]
                            ) / obj_range

            # --- selection: fill offspring from front 0 (then front 1 if needed) ---
            n_off = pop_size
            sel = []
            for front in fronts:
                if len(sel) >= n_off:
                    break
                # sort front by crowding distance descending
                front_sorted = sorted(front, key=lambda i: crowd[i], reverse=True)
                take = min(len(front_sorted), n_off - len(sel))
                sel.extend(front_sorted[:take])
            if len(sel) < n_off:
                # pad with random from non-dominated front
                nd = fronts[0]
                while len(sel) < n_off:
                    sel.append(nd[rng.integers(len(nd))])

            # --- adaptive mutation scale ---
            progress = used / max(1, self.budget)
            scale = 0.35 * (1.0 - progress) + 0.04

            # --- generate offspring ---
            off_X = np.empty((n_off, self.dim))
            for i in range(n_off):
                p = sel[rng.integers(len(sel))]
                x = X[p] + scale * span * rng.standard_normal(self.dim)
                off_X[i] = np.clip(x, lo, hi)

            off_F = np.array([problem(x) for x in off_X])
            used += n_off

            # --- environmental selection: merge and pick best pop_size ---
            X_all = np.vstack([X, off_X])
            F_all = np.vstack([F, off_F])

            rem = list(range(len(F_all)))
            all_fronts = []
            while rem:
                mask = dominated_mask(F_all[rem])
                front = [rem[i] for i in range(len(rem)) if not mask[i]]
                all_fronts.append(front)
                rem = [r for r in rem if r not in set(front)]

            keep = []
            for front in all_fronts:
                if len(keep) + len(front) <= pop_size:
                    keep.extend(front)
                else:
                    # need to trim this front by crowding
                    n_keep = np.zeros(len(F_all))
                    for k in range(self.n_obj):
                        order = np.argsort(F_all[front, k], kind='mergesort')
                        n_keep[front[order[0]]] = np.inf
                        n_keep[front[order[-1]]] = np.inf
                        obj_range = F_all[front[order[-1]], k] - F_all[front[order[0]], k]
                        if obj_range > 1e-12:
                            for j in range(1, len(order) - 1):
                                n_keep[front[order[j]]] += (
                                    F_all[front[order[j + 1]], k] - F_all[front[order[j - 1]], k]
                                ) / obj_range
                    need = pop_size - len(keep)
                    front_sorted = sorted(front, key=lambda i: n_keep[i], reverse=True)
                    keep.extend(front_sorted[:need])
                    break

            X = X_all[keep]
            F = F_all[keep]

        return X
# EVOLVE-BLOCK-END
