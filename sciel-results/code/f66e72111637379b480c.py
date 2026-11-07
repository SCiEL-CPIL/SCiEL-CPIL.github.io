"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Deterministic multi-start coordinate-descent + Nelder-Mead local search
    with an explicit wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cd_step(self, x, step, func):
        """One pass of cyclic coordinate descent. Returns (new_x, new_fx, improved)."""
        fx = func(x)
        improved = False
        for d in range(self.dim):
            base = x[d]
            for delta in (step, -step):
                cand = x.copy()
                cand[d] = np.clip(base + delta, -5.0, 5.0)
                if np.array_equal(cand, x):
                    continue
                fc = func(cand)
                if fc < fx:
                    x = cand
                    fx = fc
                    improved = True
        return x, fx, improved

    def _nm(self, x0, func, deadline, best, calls):
        """Nelder-Mead local search from x0, updating shared best/calls in place."""
        dim = self.dim
        x0 = np.clip(np.asarray(x0, dtype=float), -5.0, 5.0)
        pts = [x0.copy()]
        fx = func(x0)
        calls[0] += 1
        for i in range(dim):
            p = x0.copy()
            p[i] = x0[i] + 0.25
            p = np.clip(p, -5.0, 5.0)
            pts.append(p)
            calls[0] += 1
            if calls[0] >= self.budget:
                break
        # order by f
        order = sorted(range(len(pts)), key=lambda i: func(pts[i]) if i == 0 else self._f(pts[i], func, calls))
        # simpler: evaluate all then sort
        vals = [func(p) for p in pts]
        calls[0] += len(pts)
        idx = sorted(range(len(pts)), key=lambda i: vals[i])
        pts = [pts[i] for i in idx]
        vals = [vals[i] for i in idx]
        best_pt = pts[0]
        best_val = vals[0]
        if best_val < best[1]:
            best[0], best[1] = best_pt, best_val
        lam, gam, rho = 1.0, 0.5, -1.0
        for _ in range(400):
            if calls[0] >= self.budget or time.time() > deadline:
                break
            centroid = np.mean(pts[:-1], axis=0)
            xr = centroid + lam * (centroid - pts[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr); calls[0] += 1
            if fr < vals[0]:
                xe = centroid + gam * (xr - centroid)
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe); calls[0] += 1
                if fe < fr:
                    pts[-1] = xe; vals[-1] = fe
                else:
                    pts[-1] = xr; vals[-1] = fr
            elif fr < vals[-2]:
                pts[-1] = xr; vals[-1] = fr
            else:
                xc = centroid + rho * (centroid - pts[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = func(xc); calls[0] += 1
                if fc < vals[-1]:
                    pts[-1] = xc; vals[-1] = fc
                else:
                    for i in range(1, len(pts)):
                        pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                        pts[i] = np.clip(pts[i], -5.0, 5.0)
                        vals[i] = func(pts[i]); calls[0] += 1
                        if calls[0] >= self.budget:
                            break
            idx = sorted(range(len(pts)), key=lambda i: vals[i])
            pts = [pts[i] for i in idx]
            vals = [vals[i] for i in idx]
            if vals[0] < best[1]:
                best[0], best[1] = pts[0].copy(), vals[0]
            if calls[0] >= self.budget:
                break
        return best

    def _f(self, p, func, calls):
        v = func(p)
        calls[0] += 1
        return v

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin under 300s
        calls = [0]
        best_x = np.zeros(dim)
        best_fx = float('inf')

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_x, best_fx = np.asarray(x, dtype=float).copy(), fx

        def guarded(x):
            if calls[0] >= budget:
                return best_fx
            v = func(np.asarray(x, dtype=float))
            calls[0] += 1
            return v

        # Deterministic multi-start set: center + fixed offsets.
        starts = [np.zeros(dim)]
        offsets = [
            1.5, -1.5, 2.5, -2.5, 3.0, -3.0,
        ]
        # a handful of deterministic offset vectors (kept small and fixed)
        base_offsets = [
            np.array([o, 0, 0, 0, 0]),
            np.array([0, o, 0, 0, 0]),
            np.array([o, o, 0, 0, 0]),
            np.array([0, 0, o, o, 0]),
            np.array([o, -o, o, -o, o]),
        ]
        for o in offsets:
            for bo in base_offsets:
                s = np.zeros(dim)
                for i in range(dim):
                    s[i] = o * bo[i] if bo[i] != 0 else 0.0
                s = np.clip(s, -5.0, 5.0)
                starts.append(s)
        # de-duplicate
        seen = set()
        uniq = []
        for s in starts:
            key = tuple(np.round(s, 6))
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        starts = uniq

        # Phase 1: cheap coordinate-descent refinement from each start (bounded).
        for s in starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            x = np.clip(s.copy(), -5.0, 5.0)
            fx = guarded(x)
            record(x, fx)
            step = 1.0
            for _ in range(30):
                if calls[0] >= budget or time.time() > deadline:
                    break
                x, fx, improved = self._cd_step(x, step, guarded)
                record(x, fx)
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
            # Phase 2: Nelder-Mead polish from the best start point.
            if calls[0] >= budget or time.time() > deadline:
                break
            self._nm(x, guarded, deadline, [best_x, best_fx], calls)

        return best_x, best_fx
# EVOLVE-BLOCK-END
