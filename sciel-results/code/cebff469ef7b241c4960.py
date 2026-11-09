"""Multi-objective optimizer for RE21: HV-driven evolutionary search + adaptive mutation + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hypervolume-driven multi-objective search with adaptive mutation and Nelder-Mead local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin under the 300 s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim

        used = 0
        X = []
        F = []

        def eval_pt(x):
            nonlocal used
            used += 1
            return np.asarray(problem(x), dtype=float)

        def dominated_mask(Fv):
            """mask[i] = True if Fv[i] is dominated by some other row."""
            n = len(Fv)
            dom = np.zeros(n, dtype=bool)
            for i in range(n):
                if dom[i]:
                    continue
                d = (Fv <= Fv[i]) & (Fv < Fv[i])
                if d.sum() > 1:  # at least one other row dominates i
                    dom[i] = True
            return dom

        def compute_hv_contrib(Fv, nd_idx, ref):
            """Compute HV contribution for non-dominated points."""
            nd = Fv[nd_idx]
            n_nd = len(nd)
            hv_contrib = np.zeros(n_nd)
            if n_nd == 0:
                return hv_contrib
            order = np.argsort(nd[:, 0])
            nd_sorted = nd[order]
            prev_f1 = ref[1]
            for k in range(n_nd - 1, -1, -1):
                f0, f1 = nd_sorted[k]
                if f0 < ref[0] and f1 < ref[1]:
                    contrib = (ref[0] - f0) * (prev_f1 - f1)
                    hv_contrib[order[k]] = max(contrib, 0.0)
                else:
                    hv_contrib[order[k]] = 0.0
                prev_f1 = f1
            return hv_contrib

        # 1) Random initialization with boundary focus
        n_init = 20
        rng = np.random.default_rng(12345)
        for i in range(n_init):
            if i < n_init // 2:
                x = lo + span * rng.random(dim)
            else:
                # Boundary-focused: force some dims to be near bounds
                u = rng.random(dim)
                mask = rng.random(dim) < 0.3
                if np.any(mask):
                    u[mask] = rng.random(np.sum(mask))
                x = lo + span * u
            X.append(x)
            F.append(eval_pt(x))
        if used >= self.budget:
            return np.array(X)

        # 2) Evolutionary loop with adaptive sigma
        sigma = 0.15
        min_sigma = 0.01
        max_sigma = 0.3
        stagnation = 0
        best_hv = 0.0
        ref = np.array([3000.0, 0.06])

        while used < self.budget:
            if time.time() > deadline - 20.0:  # reserve time for polish
                break

            Fv = np.array(F)
            Xv = np.array(X)
            dom = dominated_mask(Fv)
            nd_idx = np.where(~dom)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(Fv))

            # Compute HV contribution (2D) for each non-dominated point
            hv_contrib = compute_hv_contrib(Fv, nd_idx, ref)

            total_hv = hv_contrib.sum()
            if total_hv > best_hv:
                best_hv = total_hv
                stagnation = 0
            else:
                stagnation += 1

            # Adaptive sigma: grow if stagnant, shrink if improving
            if stagnation > 30:
                sigma = min(max_sigma, sigma * 1.5)
                stagnation = 0
            elif total_hv > best_hv * 0.95:
                sigma = max(min_sigma, sigma * 0.97)

            # Selection: HV-contribution driven
            if total_hv <= 0:
                p = rng.integers(len(nd_idx))
            else:
                probs = hv_contrib / total_hv
                # Add small epsilon to avoid zero-probability issues
                probs = probs + 1e-10
                probs /= probs.sum()
                p = rng.choice(len(nd_idx), p=probs)

            parent = Xv[nd_idx[p]].copy()

            # Mutation
            step = sigma * span * rng.standard_normal(dim)
            child = np.clip(parent + step, lo, hi)
            fc = eval_pt(child)

            # Replace the point with the lowest HV contribution
            worst = int(np.argmin(hv_contrib))
            X[nd_idx[worst]] = child
            F[nd_idx[worst]] = fc

        # 3) Local polish: Nelder-Mead on top HV-contributing points
        if time.time() < deadline - 5.0 and used < self.budget:
            Fv = np.array(F)
            Xv = np.array(X)
            dom = dominated_mask(Fv)
            nd_idx = np.where(~dom)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(Fv))

            hv_contrib = compute_hv_contrib(Fv, nd_idx, ref)
            n_nd = len(nd_idx)

            # Use top 5 HV contributors as starting points for Nelder-Mead
            top_k = min(5, n_nd, max(1, (self.budget - used) // 2))
            top_indices = np.argsort(hv_contrib)[::-1][:top_k]

            # Weight vectors for weighted-sum (tuned for diversity)
            n_w = min(8, max(1, (self.budget - used) // top_k))
            weights = np.array([[1.0 - i / max(n_w - 1, 1), i / max(n_w - 1, 1)]
                                for i in range(n_w)])

            for w in weights:
                if used >= self.budget or time.time() > deadline - 2.0:
                    break

                def obj_func(x, w=w):
                    f = np.asarray(problem(x), dtype=float)
                    return float(np.dot(w, f))

                # For each weight, find best starting point among top-k
                Fv2 = np.array(F)
                scores = Fv2[nd_idx] @ w
                best_local = nd_idx[int(np.argmin(scores))]
                x0 = Xv[best_local].copy()

                try:
                    res = minimize(obj_func, x0, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8,
                                            'adaptive': True})
                    if res.success and used < self.budget:
                        xf = np.clip(res.x, lo, hi)
                        ff = eval_pt(xf)
                        # Replace worst HV point
                        Fv3 = np.array(F)
                        dom3 = dominated_mask(Fv3)
                        nd3 = np.where(~dom3)[0]
                        if len(nd3) > 0:
                            hv3 = compute_hv_contrib(Fv3, nd3, ref)
                            worst3 = int(np.argmin(hv3))
                            X[nd3[worst3]] = xf
                            F[nd3[worst3]] = ff
                except Exception:
                    pass

        return np.array(X)
# EVOLVE-BLOCK-END
