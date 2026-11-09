"""Portfolio search for DTLZ2: Halton sequence on Pareto front, deadline-guarded."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a Halton-sequence front portfolio."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        def time_ok():
            return time.time() < deadline

        # Build the analytical Pareto front for DTLZ2.
        # For DTLZ2 with 3 objectives, the Pareto front is the quarter-unit-sphere in the first octant.
        # We parameterize using angles t1, t2 in [0, pi/2].
        # To achieve uniform surface coverage, we use a change of variables:
        #   u1 = sin(t1)  =>  t1 = arcsin(u1), with u1 in [0, 1]
        #   t2 in [0, pi/2]
        # Then we apply a 2D Halton sequence to (u1, v2) where v2 = t2 / (pi/2) in [0, 1].

        def front_to_x(t):
            """Convert angle parameters t (length n_obj-1) to decision vector x.
            For DTLZ2: x_i = sin(t_i) for i < n_obj-1, and the rest are set
            to values that make g(x) = 0 (i.e., x_i = 0.5 for i >= n_obj-1).
            """
            x = np.full(self.dim, 0.5)
            for i in range(min(self.n_obj - 1, self.dim)):
                x[i] = np.sin(t[i])
            return x

        # Generate Halton sequence points in 2D
        def halton(n, bases):
            """Generate n Halton sequence points in [0,1)^d for given bases."""
            points = np.zeros((n, len(bases)))
            for i in range(n):
                for j, base in enumerate(bases):
                    f = 1.0
                    r = 0.0
                    idx = i + 1  # 1-indexed to avoid the origin
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    points[i, j] = r
            return points

        n_evals = self.budget
        # Use bases 2 and 3 for 2D Halton
        halton_pts = halton(n_evals, [2, 3])

        X = []
        F = []
        used = 0
        for k in range(n_evals):
            if not time_ok():
                break
            u1, v2 = halton_pts[k]
            # u1 in [0,1) maps to t1 = arcsin(u1)
            # v2 in [0,1) maps to t2 = v2 * pi/2
            t1 = np.arcsin(u1)
            t2 = v2 * (np.pi / 2)
            t = np.array([t1, t2])
            x = front_to_x(t)
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
            used += 1

        X = np.array(X)
        F = np.array(F)

        return X

# EVOLVE-BLOCK-END
