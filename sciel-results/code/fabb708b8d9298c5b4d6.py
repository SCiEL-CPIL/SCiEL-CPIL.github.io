"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective search: random init + DE mutation, tournament selection,
    and hypervolume-informed replacement, guarded by an explicit deadline."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim, budget, n_obj = self.dim, self.budget, self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random  # evaluator has already seeded np.random

        # ---- population size (small: keeps per-iteration cost low) ----
        pop = int(min(16, max(4, budget // 10)))

        # ---- initial random population ----
        X = lo + span * rng.random((pop, dim))
        F = np.array([problem(x) for x in X])
        used = pop

        # ---- reference point for hypervolume-informed eviction ----
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # ---- fast non-dominated sort (returns front index per point) ----
        def fronts(f):
            n = len(f)
            rank = np.full(n, -1, dtype=int)
            done = 0
            remaining = set(range(n))
            while remaining:
                front = []
                for i in list(remaining):
                    fi = f[i]
                    dom = np.any(np.all(f <= fi, axis=1) & np.any(f < fi, axis=1))
                    if not dom:
                        front.append(i)
                for i in front:
                    rank[i] = done
                done += 1
                remaining.difference_update(front)
            return rank

        # ---- hypervolume contribution of a point (2D projection over obj0, obj1) ----
        def hv_contrib(i, f):
            fi = f[i]
            # points strictly better than fi in obj0 and obj1, and strictly better than ref
            mask = (f[:, 0] < fi[0]) & (f[:, 1] < fi[1]) & (f[:, 0] < ref[0]) & (f[:, 1] < ref[1])
            if not np.any(mask):
                return (ref[0] - fi[0]) * (ref[1] - fi[1])
            # points that dominate fi (obj0<=fi0 and obj1<=fi1, at least one strict)
            dom = (f[:, 0] <= fi[0]) & (f[:, 1] <= fi[1]) & (
                (f[:, 0] < fi[0]) | (f[:, 1] < fi[1]))
            # contribution region: x in [fi0, ref0], y in [fi1, ref1], minus dominated sub-rects
            # approximate by sampling the bounding box of the dominated region
            if np.any(dom):
                dx = f[dom, 0].min()
                dy = f[dom, 1].min()
                if dx < fi[0] and dy < fi[1]:
                    # subtract the rectangle [dx,fi0]x[dy,fi1] from [fi0,ref0]x[fi1,ref1]?
                    # The dominated region lies below-left of fi; it does not overlap fi's own box.
                    # So fi's contribution in the (obj0,obj1) plane is simply its box.
                    pass
            return (ref[0] - fi[0]) * (ref[1] - fi[1])

        def pick_worst(f):
            n = len(f)
            # cheap: prefer points worse than ref in some objective
            worse = np.any(f >= ref, axis=1)
            if np.any(worse):
                return int(np.argmax(np.sum(np.where(f >= ref, f, 0.0), axis=1)))
            # otherwise evict smallest hypervolume contribution (2D proxy)
            contribs = np.array([hv_contrib(i, f) for i in range(n)])
            return int(np.argmin(contribs))

        # ---- main loop: DE mutation + tournament selection + HV-informed replacement ----
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin well under 330s
        cr = 0.9
        F_mut = 0.8

        while used < budget:
            if time.time() > deadline:
                break

            rank = fronts(F)

            # tournament: pick 3, choose lowest rank (best front), break ties by random
            t = np.sort(rng.integers(0, pop, size=3))
            tr = rank[t]
            best = t[np.argmin(tr)]

            # DE mutation: base + F*(a - b)
            a = int(rng.integers(pop))
            b = int(rng.integers(pop))
            while a == b or a == best or b == best:
                a = int(rng.integers(pop))
                b = int(rng.integers(pop))

            v = X[best] + F_mut * (X[a] - X[b])
            # polynomial-style mutation: add small random perturbation
            v = v + 0.05 * span * rng.standard_normal(dim)
            v = np.clip(v, lo, hi)

            fv = problem(v)
            used += 1

            # replace worst point (hypervolume-informed)
            w = pick_worst(F)
            X[w] = v
            F[w] = fv

        return X
# EVOLVE-BLOCK-END
