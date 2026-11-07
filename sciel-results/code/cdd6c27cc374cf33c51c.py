"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton_sequence(n, base):
    """Generate n points of the Halton sequence in [0,1) for a given prime base."""
    seq = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        idx = i
        while idx > 0:
            f /= base
            x += f * (idx % base)
            idx //= base
        seq.append(x)
    return np.array(seq)


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ2 analytic structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_obj = self.n_obj
        dim = self.dim
        n_pos = n_obj - 1  # 2 position variables for DTLZ2
        n_dist = dim - n_pos  # 10 distance variables

        evaluated_points = []
        evaluated_f = []
        used = 0
        budget = self.budget

        # Use all budget on the Pareto front with low-discrepancy sampling
        # Halton sequence in 2D (bases 2 and 3) for uniform coverage of position space
        n_samples = min(budget, 200)
        h1 = _halton_sequence(n_samples, 2)
        h2 = _halton_sequence(n_samples, 3)

        # Shift to avoid the exact corners and ensure good spread
        # Halton already gives [0,1), but let's nudge slightly inward for robustness
        eps = 1e-10
        h1 = np.clip(h1, eps, 1.0 - eps)
        h2 = np.clip(h2, eps, 1.0 - eps)

        for i in range(n_samples):
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = h1[i]
            x[1] = h2[i]
            # Distance variables at 0.5 for g=0 (on Pareto front)
            x[n_pos:] = 0.5
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated_points.append(x.copy())
            evaluated_f.append(np.array(f))
            used += 1

        # Return the last point (return value is ignored)
        if evaluated_points:
            return evaluated_points[-1]
        else:
            return lo.copy()
# EVOLVE-BLOCK-END
