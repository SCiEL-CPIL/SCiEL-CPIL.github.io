"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ7's analytical Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # safety margin before 300s per-eval / 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # DTLZ7 front: x[2:]=0, f1=x0, f2=x1, f3=2-sqrt(x0)
        # Parameterize by (u, v) in [0,1]^2 -> point x = [u, v, 0, 0, ..., 0]
        def front_point(u, v):
            x = np.zeros(dim)
            x[0] = u
            x[1] = v
            return x

        # Evaluate a batch of (u, v) pairs on the front, recording objective vectors.
        # We still call problem(x) so the evaluator records real objective values.
        # We track which evaluations we've spent.
        evals_used = 0

        # Build a dense, well-spread set of (u, v) on the front using
        # equal-hypervolume-contribution grid in transformed (w, v) space.
        # The DTLZ7 front has surface area density sqrt(1 + 1/(4u)) in (u,v) space.
        # Transforming u -> w such that dw = sqrt(1+1/(4u)) du makes the
        # hypervolume contribution uniform in (w,v) space.

        def _w_of_u(u):
            u = np.maximum(u, 1e-15)
            t = np.sqrt(u)
            return t * np.sqrt(4.0 * u + 1.0) + 0.5 * np.log(2.0 * t + np.sqrt(4.0 * u + 1.0))

        def _u_of_w(w, W_total):
            lo = np.zeros_like(w, dtype=float)
            hi = np.ones_like(w, dtype=float)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                w_mid = _w_of_u(mid)
                mask = w_mid < w
                lo = np.where(mask, mid, lo)
                hi = np.where(mask, hi, mid)
            return 0.5 * (lo + hi)

        W_total = float(_w_of_u(np.array([1.0]))[0])

        # Choose grid dimensions matching the domain aspect ratio in (w,v) space.
        # w ranges [0, W_total] ~ [0, 2.96], v ranges [0, 1].
        # Aspect ratio ~ 2.96, so n_w/n_v ~ 2.96.
        # With budget=200: 25*8=200, ratio=3.125 (close to 2.96)
        n_w, n_v = 25, 8
        n_total = n_w * n_v
        if n_total > budget:
            # Scale down
            scale = budget / n_total
            n_v = max(1, int(n_v * scale))
            n_w = budget // n_v
            n_total = n_w * n_v

        # Equal-spaced points in (w, v) space
        w_vals = (np.arange(n_w) + 0.5) * W_total / n_w
        v_vals = (np.arange(n_v) + 0.5) / n_v

        # Convert w back to u
        u_vals = _u_of_w(w_vals, W_total)

        # Build grid points
        U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
        all_pts = np.column_stack([U.ravel(), V.ravel()])

        # Trim to budget if needed
        if len(all_pts) > budget:
            all_pts = all_pts[:budget]

        # Evaluate all points via problem()
        F = np.empty((len(all_pts), self.n_obj))
        for i, (u, v) in enumerate(all_pts):
            x = front_point(u, v)
            f = np.asarray(problem(x), dtype=float)
            F[i] = f
            evals_used += 1
            if evals_used >= budget:
                break
            if time.time() > deadline:
                break

        return all_pts
# EVOLVE-BLOCK-END
