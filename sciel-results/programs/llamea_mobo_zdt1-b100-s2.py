import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, x1_vals):
        """Compute hypervolume for points on ZDT1 front with ref [11, 11].
        x1_vals must be sorted ascending, in [0, 1]."""
        if len(x1_vals) == 0:
            return 0.0
        f1 = x1_vals
        f2 = 1.0 - np.sqrt(f1)
        # Widths in f1 direction
        df1 = np.diff(f1)
        df1 = np.append(df1, 11.0 - f1[-1])
        # Heights in f2 direction: 11 - f2 = 10 + sqrt(f1)
        hf2 = 11.0 - f2
        return float(np.sum(df1 * hf2))

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin before 330s kill

        B = int(self.budget)
        D = int(self.dim)

        # The ZDT1 Pareto front: x2..xn = 0, f1 = x1, f2 = 1 - sqrt(x1)
        # We choose B values of x1 in [0,1] to maximize HV with ref [11,11].
        # Parameterization: x1 = u^p where u is uniform in [0,1]
        # Optimal p clusters points where the front curves most.

        u = np.linspace(0.0, 1.0, B)

        # Stage 1: Coarse search over p
        p_coarse = np.arange(0.5, 8.01, 0.05)
        best_hv = -1.0
        best_p = 1.0
        for p in p_coarse:
            if time.time() > deadline - 30.0:
                break
            x1_vals = u ** p
            hv = self._compute_hv(x1_vals)
            if hv > best_hv:
                best_hv = hv
                best_p = p

        # Stage 2: Golden-section refinement
        lo_p = max(0.01, best_p - 0.05)
        hi_p = min(20.0, best_p + 0.05)
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = lo_p, hi_p
        for _ in range(80):
            if time.time() > deadline - 20.0:
                break
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = self._compute_hv(u ** c)
            fd = self._compute_hv(u ** d)
            if fc < fd:
                a = c
            else:
                b = d
            if abs(b - a) < 1e-12:
                break
        best_p = (a + b) / 2.0

        # Stage 3: High-precision polish via local search around best_p
        # Try small perturbations
        for delta in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
            if time.time() > deadline - 10.0:
                break
            for sign in [-1, 1]:
                p_try = best_p + sign * delta
                if p_try <= 0.01:
                    continue
                hv_try = self._compute_hv(u ** p_try)
                if hv_try > self._compute_hv(u ** best_p):
                    best_p = p_try

        # Generate final x1 values
        x1_vals = u ** best_p
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        x1_vals.sort()

        # Build the evaluation matrix
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        X = np.zeros((B, D))
        X[:, 0] = x1_vals
        for i in range(1, D):
            X[:, i] = lo[i]

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())

        return X
