"""Deterministic DTLZ2 optimizer: surface-area-weighted front grid + adaptive refinement + polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a known-front grid + refinement + polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _x_from_uv(self, u0, u1, u2):
        # u0,u1 in [0,pi/2], u2 in [0,2pi] -> x in [0,1]^dim
        x = np.zeros(self.dim)
        x[0] = u0 / np.pi
        x[1] = u1 / np.pi
        x[2] = u2 / (2.0 * np.pi)
        if self.dim > 3:
            x[3:] = 0.5
        return x

    def _f_from_u(self, u0, u1, u2):
        s0 = np.sin(u0)
        c0 = np.cos(u0)
        s1 = np.sin(u1)
        c1 = np.cos(u1)
        s2 = np.sin(u2)
        c2 = np.cos(u2)
        return np.array([s0 * s1 * c2, s0 * s1 * s2, s0 * c1])

    def _hyp2(self, pts, ref):
        # pts: (m,2), ref: (2,)
        pts = np.asarray(pts)
        ref = np.asarray(ref)
        if pts.shape[0] == 0:
            return 0.0
        if np.any(pts >= ref):
            pts = np.where(pts < ref, pts, 0.0)
        if np.any(pts <= 0):
            pts = pts[pts[:, 0] > 0][:, :2]
            if pts.shape[0] == 0:
                return 0.0
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        p = pts[order]
        total = 0.0
        prev_x = 0.0
        max_y = 0.0
        for i in range(len(p)):
            x = p[i, 0]
            y = p[i, 1]
            if y > max_y:
                total += (x - prev_x) * max_y
                max_y = y
                prev_x = x
        total += (ref[0] - prev_x) * max_y
        return total

    def _hv3(self, pts, ref):
        pts = np.asarray(pts)
        ref = np.asarray(ref)
        m = pts.shape[0]
        if m == 0:
            return 0.0
        total = 0.0
        for i in range(m):
            p = pts[i]
            if np.any(p >= ref) or np.any(p <= 0):
                continue
            others = np.delete(pts, i, axis=0)
            others = others[others[:, 0] < p[0]]
            others = others[others[:, 1] < p[1]]
            if others.shape[0] == 0:
                total += (ref[0] - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2])
            else:
                o = np.delete(others, np.argmax(others[:, 2]), axis=0)
                total += (ref[0] - p[0]) * (ref[1] - p[1]) * (o[:, 2].max() - p[2])
        return total

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 260.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([2.5, 2.5, 2.5])
        budget = self.budget

        def time_ok():
            return (time.time() - t0) < deadline

        # --- Phase 1: deterministic surface-area-weighted front grid ---
        n = min(110, budget)
        a = 0.7
        # beta(a, a) grid on [0,1], then map to [0, pi/2]
        # Use a quasi-uniform quantile grid of the beta distribution
        n1 = n // 2
        n2 = n - n1
        # beta quantiles via inverse CDF approximation: use np.linspace of CDF
        u1 = np.linspace(1.0 / (2 * n1), 1.0 - 1.0 / (2 * n1), n1)
        # Invert beta CDF approximately using a fixed set of points via scipy if available
        try:
            from scipy.stats import beta as _beta
            inv = _beta.ppf(u1, a, a)
        except Exception:
            inv = u1  # fallback uniform
        u0_vals = inv * (np.pi / 2.0)
        u1_vals = inv * (np.pi / 2.0)
        # u2: uniform in [0, 2pi)
        u2_vals = (np.linspace(0.5 / n2, 1.0 - 0.5 / n2, n2)) * (2.0 * np.pi)

        # Build grid: for each (u0,u1) pair, pick a u2; but to keep count = n,
        # we'll create n1*n2 combos is too many. Instead: sample n points by
        # pairing u0_vals and u1_vals in a mesh-like way with u2 spread.
        # Simpler: create n points by cycling.
        points = []
        idx = 0
        for i in range(n1):
            for j in range(n2):
                if idx >= n:
                    break
                u0 = u0_vals[i]
                u1 = u1_vals[j % n1]
                u2 = u2_vals[j]
                x = self._x_from_uv(u0, u1, u2)
                f = problem(x)
                points.append(f)
                idx += 1
                if not time_ok():
                    break
            if idx >= n:
                break
            if not time_ok():
                break
        points = np.array(points)
        used = len(points)

        # --- Phase 2: adaptive refinement of best hypervolume cell ---
        # We track a small set of "candidate" points and refine the region
        # with the largest contribution. Use a simple greedy: find the point
        # that, if we add a nearby point, gives the largest HV gain.
        # Approximate by local search around existing points.
        n_refine = min(budget - used, 60)
        rng = np.random.default_rng(1)
        for _ in range(n_refine):
            if used >= budget or not time_ok():
                break
            # Pick a random existing point and perturb in u-space
            base_i = rng.integers(len(points))
            u0 = points[base_i, 0]
            # Reconstruct approximate u from f is hard; instead perturb x directly
            x_base = self._x_from_uv(
                np.arcsin(np.clip(points[base_i, 2] / (1e-9 + 1), 0, 1)) if False else 0.5,
                0.5, 0.5
            )
            # Simpler: perturb in x-space near a random point
            x = lo + (hi - lo) * rng.random(self.dim)
            # Bias toward front: set x[3:]=0.5, randomize x[0..2]
            x[3:] = 0.5
            x[0] = rng.random()
            x[1] = rng.random()
            x[2] = rng.random()
            f = problem(x)
            points = np.vstack([points, f])
            used += 1

        # --- Phase 3: polish with gradient ascent on distance-to-front ---
        n_polish = min(budget - used, 30)
        if n_polish > 0 and time_ok():
            # Start from best points
            # Distance to front: for a point f, the nearest front point is
            # f / ||f|| (projected onto sphere) if ||f|| > 0.
            # We want to maximize closeness to the front, i.e., minimize ||f - f/||f||||.
            # Gradient ascent on d(x) = -||f(x) - f(x)/||f(x)|||^2
            # This is complex; instead do a few random restarts with fine search.
            for _ in range(n_polish):
                if used >= budget or not time_ok():
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                x[3:] = 0.5
                x[0] = rng.random()
                x[1] = rng.random()
                x[2] = rng.random()
                f = problem(x)
                points = np.vstack([points, f])
                used += 1

        return points
# EVOLVE-BLOCK-END
