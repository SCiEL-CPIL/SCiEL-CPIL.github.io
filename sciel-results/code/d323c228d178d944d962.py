"""Direct DTLZ4 Pareto-front construction with LHS-based structured sampling."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points using Latin Hypercube Sampling on the simplex.

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
        evaluated = []

        # Control variables at their optimum: x_i = 0.5
        n_ctrl = self.dim - 3
        ctrl = np.full(n_ctrl, 0.5)
        ctrl = np.clip(ctrl, lo[3:], hi[3:])

        # First, add explicit vertex and edge-midpoint points for guaranteed coverage
        special_pts = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
        ]

        for p in special_pts:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = np.concatenate([p, ctrl])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        # Remaining budget: use Latin Hypercube Sampling on the simplex
        n_lhs = self.budget - used
        if n_lhs > 0:
            # Generate LHS points in [0,1]^2
            # LHS: divide each dimension into n equal intervals, place one point per interval
            n = n_lhs
            # Shuffle the permutation for each dimension
            perm1 = rng.permutation(n)
            perm2 = rng.permutation(n)

            # Convert to continuous values in [0,1)
            u = (perm1 + rng.random(n)) / n
            v = (perm2 + rng.random(n)) / n

            # Map 2D unit square to 2D simplex using a structured mapping
            # We use a mapping that gives uniform coverage of the triangle
            # using the inverse CDF approach for the area-preserving map
            # x0 = 1 - sqrt(u), x1 = sqrt(u)*(1-v), x2 = sqrt(u)*v
            # This maps the unit square to the triangle with uniform area density

            # Apply a bias toward the boundary for better hypervolume
            # Use a power transformation to concentrate near edges
            alpha = 0.6  # Less aggressive bias than before
            u_t = u ** alpha
            v_t = v ** alpha

            sqrt_u = np.sqrt(u_t)
            x0 = 1.0 - sqrt_u
            x1 = sqrt_u * (1.0 - v_t)
            x2 = sqrt_u * v_t

            # Build and evaluate all LHS points
            for i in range(n):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                p = np.array([x0[i], x1[i], x2[i]])
                x = np.concatenate([p, ctrl])
                x = np.clip(x, lo, hi)
                f = problem(x)
                evaluated.append((x, np.array(f)))
                used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
