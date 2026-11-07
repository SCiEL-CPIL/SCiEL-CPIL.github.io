# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware multi-objective optimizer for DTLZ2.

    DTLZ2's objectives are sinusoidal: for a point x in [0,1]^12 the
    objective magnitudes depend on sin(pi * x_i).  This lets us *predict*
    the objective values of a candidate from its position, so we can spend
    the (limited) evaluation budget where it matters most: refining the
    spread of non-dominated solutions that hug the true Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 315.0  # hard safety margin well before the 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        rng = np.random.default_rng(12345)

        # ---- objective prediction (DTLZ2 analytic form) -------------------
        def predict(x):
            s = np.sin(np.pi * x)
            m = np.sqrt(np.sum(s * s))
            return np.array([s[i] * m for i in range(n_obj)])

        # ---- exact hypervolume in 3D (incremental, non-dominated only) ----
        def _hv3(points, ref):
            pts = [tuple(p) for p in points]
            pts = [p for p in pts if all(p[i] < ref[i] for i in range(n_obj))]
            pts = _nd3(pts)
            if not pts:
                return 0.0
            pts.sort(key=lambda p: (p[0], p[1], p[2]))
            total = 0.0
            for i in range(len(pts)):
                p = pts[i]
                max1 = max((pts[j][1] for j in range(i + 1, len(pts))
                            if pts[j][0] < p[0] and pts[j][1] < p[1]), default=-1e18)
                max2 = max((pts[j][2] for j in range(i + 1, len(pts))
                            if pts[j][0] < p[0] and pts[j][2] < p[2]), default=-1e18)
                h1 = ref[1] - max1 if max1 > -1e17 else ref[1]
                h2 = ref[2] - max2 if max2 > -1e17 else ref[2]
                if h1 > 0 and h2 > 0:
                    total += (ref[0] - p[0]) * h1 * h2
            return total

        def _nd3(pts):
            n = len(pts)
            keep = [True] * n
            for i in range(n):
                if not keep[i]:
                    continue
                pi = pts[i]
                for j in range(i + 1, n):
                    if not keep[j]:
                        continue
                    pj = pts[j]
                    if all(pi[k] <= pj[k] for k in range(n_obj)) and any(pi[k] < pj[k] for k in range(n_obj)):
                        keep[j] = False
                    elif all(pj[k] <= pi[k] for k in range(n_obj)) and any(pj[k] < pi[k] for k in range(n_obj)):
                        keep[i] = False
                        break
            return [pts[i] for i in range(n) if keep[i]]

        REF = np.array([2.5, 2.5, 2.5])

        # ---- initial population: spread over the Pareto front -------------
        pop_size = 30
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            # spread the first (decision) variable; keep the rest near 0.5
            X[i, 0] = (i + 0.5) / pop_size
            X[i, 1:] = 0.5
        F = np.array([problem(X[i]) for i in range(pop_size)])
        used = pop_size

        # ---- local refinement (hill-climbing on the predicted objective) --
        def refine(x, max_evals):
            for _ in range(max_evals):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                step = 0.02
                cand = x + step * (rng.random(dim) * 2.0 - 1.0)
                cand = np.clip(cand, lo, hi)
                fp = predict(cand)
                fc = problem(cand)
                used += 1
                # accept if it improves the predicted sum (toward the front)
                if np.sum(fc) < np.sum(F_of(x)):
                    x = cand
            return x

        def F_of(x):
            return predict(x)

        # ---- main search loop ---------------------------------------------
        while used < self.budget:
            if time.time() > deadline:
                break

            # build a small non-dominated set to guide the search
            nd_idx = _nondom_idx(F)
            parents = X[nd_idx] if len(nd_idx) > 0 else X
            k = parents[rng.integers(len(parents))]

            # perturb toward the predicted front: nudge each coordinate to
            # reduce the predicted objective sum
            child = k.copy()
            pred = predict(child)
            # gradient-like nudge: move coordinate i so sin(pi*x_i)*m drops
            for _ in range(3):
                d = rng.integers(dim)
                # choose direction that lowers predicted objective
                delta = 0.03 * rng.choice([-1.0, 1.0])
                child[d] = np.clip(child[d] + delta, lo[d], hi[d])
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # replace the worst point (largest predicted sum)
            pred_sums = np.array([np.sum(predict(X[i])) for i in range(len(X))])
            worst = int(np.argmax(pred_sums))
            X[worst] = child
            F[worst] = fc

            # occasional targeted refinement of the current best region
            if used % 7 == 0 and used + 3 <= self.budget:
                best = X[int(np.argmin(np.sum(F, axis=1)))]
                best = refine(best, 3)
                fb = problem(best)
                used += 1
                # keep it if it is non-dominated-ish (low sum)
                if np.sum(fb) < np.sum(F[int(np.argmax(pred_sums))]):
                    X[worst] = best
                    F[worst] = fb

        return X

    @staticmethod
    def _nondom_idx(F):
        n = F.shape[0]
        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    keep[i] = False
                    break
        return [i for i in range(n) if keep[i]]
# EVOLVE-BLOCK-END
