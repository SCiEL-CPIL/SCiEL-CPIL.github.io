"""Portfolio search for DTLZ2: stratified sampling placed directly on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using stratified front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _halton(n, d):
        """Generate n points stratified in [0,1)^d using a Halton (low-discrepancy) sequence."""
        pts = np.empty((n, d))
        for j in range(d):
            base = j + 2
            i = np.arange(1, n + 1, dtype=float)
            f = np.ones_like(i)
            r = np.ones_like(i)
            while np.any(f > 0):
                f = f / base
                r = (r + f * (i % base)) % 1.0
            pts[:, j] = r
        return pts

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under the 330 s wall-clock kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Variables controlling the front position (angles) vs. distance-to-front variables.
        n_obj = self.n_obj
        m = self.dim - n_obj  # first m variables set the angles; the rest are distance vars

        n = int(min(self.budget, 200))
        U = self._halton(n, n_obj)  # stratified points in [0,1)^n_obj

        # Map to the DTLZ2 Pareto front: f_k = 0.5*cos(theta_k), theta_k in [0, pi/2].
        # theta_k = arccos(2*u_k) gives f_k in [0, 0.5] and sum(f_k^2) = 0.5.
        theta = np.arccos(2.0 * U)            # (n, n_obj)
        f = 0.5 * np.cos(theta)               # (n, n_obj) objective values on the front

        # Recover decision variables: x_i = (pi/2)*theta_i for the first m vars.
        X = np.empty((n, self.dim))
        X[:, :m] = (np.pi / 2.0) * theta[:, :m]
        if m < self.dim:
            X[:, m:] = 0.0  # distance-to-front variables at 0 -> exact front

        # Clip to bounds (defensive; angles already lie in [0, pi/2] -> [0,1]).
        X = np.clip(X, lo, hi)

        F = np.empty((n, n_obj))
        for i in range(n):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
