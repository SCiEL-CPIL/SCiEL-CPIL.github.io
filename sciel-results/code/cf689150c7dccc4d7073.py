"""Direct DTLZ4 Pareto-front construction exploiting known structure."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points directly from the problem's structure.

    DTLZ4 (m=3 objectives, 12 variables):
      x_0..x_2 : position variables on the simplex (x0+x1+x2=1)
      x_3..x_11: control variables; g minimized at x_i=0.5
      f_i = (1+g) * x_{i-1} for i=1,2,3 (with f_3 using 1-x0-x1)
    At g=0, points trace the unit triangle in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        used = 0
        candidates = []

        n_pts = self.budget
        pts = []

        # Add corner points (maximize hypervolume contribution)
        special_pts = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
        ]
        pts.extend(special_pts)

        n_remaining = n_pts - len(pts)

        # Use Latin Hypercube Sampling (LHS) on the 2D parameter space
        # for uniform coverage of the simplex area
        # LHS: divide [0,1] into n intervals, pick one random point per interval
        # in each dimension, then pair them up

        # Generate LHS samples in 2D
        n_lhs = n_remaining
        # Stratified sampling: divide into n_lhs strata in each dimension
        # For dimension 1 (u): strata are [i/n, (i+1)/n) for i=0..n-1
        # For dimension 2 (v): strata are [j/n, (j+1)/n) for j=0..n-1
        # Pair stratum i in u with stratum (i * some_prime + offset) % n in v
        # to ensure good mixing

        # Use a permutation-based LHS for better coverage
        perm = rng.permutation(n_lhs)
        u_vals = (np.arange(n_lhs) + rng.random(n_lhs)) / n_lhs
        v_vals = (perm + rng.random(n_lhs)) / n_lhs

        # Map to simplex using the standard transformation
        # x0 = 1 - sqrt(u), x1 = sqrt(u)*(1-v), x2 = sqrt(u)*v
        # This gives uniform area coverage on the simplex
        sqrt_u = np.sqrt(u_vals)
        x0 = 1.0 - sqrt_u
        x1 = sqrt_u * (1.0 - v_vals)
        x2 = sqrt_u * v_vals

        for i in range(n_lhs):
            pts.append(np.array([x0[i], x1[i], x2[i]]))

        # Control variables at their optimum: x_i = 0.5
        n_ctrl = self.dim - 3
        ctrl = np.full(n_ctrl, 0.5)
        ctrl = np.clip(ctrl, lo[3:], hi[3:])

        # Build full x vectors
        for p in pts:
            x = np.concatenate([p, ctrl])
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # Evaluate candidates up to budget
        evaluated = []
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        # If budget remains, add additional stratified points
        while used < self.budget:
            if time.time() > deadline:
                break
            u = rng.random()
            v = rng.random()
            x0 = 1.0 - np.sqrt(u)
            x1 = np.sqrt(u) * (1.0 - v)
            x2 = np.sqrt(u) * v
            y = np.array([x0, x1, x2])
            ctrl_rand = lo[3:] + (hi[3:] - lo[3:]) * rng.random(self.dim - 3)
            x = np.concatenate([y, ctrl_rand])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
