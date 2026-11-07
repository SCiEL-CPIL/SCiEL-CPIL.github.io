"""Rewritten optimizer: deterministic lattice + coordinate-wise line search for f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """
    Two-phase optimizer for BBOB f18 (Schaffers F7 ill-conditioned):
      Phase 1: Coarse-to-fine deterministic lattice search to locate the basin.
      Phase 2: Coordinate-wise line search (golden-section per axis) exploiting
               the axis-aligned ill-conditioning of Schaffer's F7.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        DEADLINE = t_start + 320.0  # 10s safety margin before 330s kill

        call_count = 0

        def time_ok():
            return time.time() < DEADLINE

        def safe_call(x):
            nonlocal call_count
            if not time_ok() or call_count >= budget:
                return float('inf')
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                val = float(func(x_clipped))
                call_count += 1
                return val
            except Exception:
                call_count += 1
                return float('inf')

        # ---------- Phase 1: Coarse-to-fine lattice ----------
        # Start with a coarse grid, refine around the best point
        best_x = np.zeros(dim)
        best_f = float('inf')

        # Coarse grid: 3 points per dimension -> 3^5 = 243 evaluations
        # But we have budget 500, so let's be smarter.
        # Use a hierarchical lattice: first 2^dim, then refine.

        # Level 1: 2 points per dim (midpoints of halves) -> 2^5 = 32 evals
        # Actually, let's do a structured exploration:
        # First evaluate center and corners of the box.

        # Evaluate center
        center = np.zeros(dim)
        f_center = safe_call(center)
        if f_center < best_f:
            best_f = f_center
            best_x = center.copy()

        # Evaluate corners and edge midpoints (10 points for dim=5)
        # Actually, let's do a small set of diverse points
        rng = np.random.default_rng(42)

        # Phase 1a: Random sampling for global coverage
        n_random = min(30, budget - 1)
        for _ in range(n_random):
            if not time_ok() or call_count >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1b: Coarse lattice around best point
        # Grid with spacing 2.0, centered on best_x, clipped to [-5,5]
        for level in range(3):
            if not time_ok() or call_count >= budget:
                break
            spacing = 2.0 / (2 ** level)  # 2.0, 1.0, 0.5
            n_per_dim = max(2, int(10.0 / spacing))  # number of points per dim
            n_per_dim = min(n_per_dim, 7)  # cap to keep total evals manageable

            # Build lattice centered on best_x
            offsets = np.linspace(-n_per_dim / 2 * spacing, n_per_dim / 2 * spacing, n_per_dim)
            lattice = np.array(np.meshgrid(*[offsets] * dim), dtype=float).transpose(-1, *range(dim)).reshape(-1, dim)
            lattice = lattice + best_x
            lattice = np.clip(lattice, -5.0, 5.0)

            # Evaluate lattice points (skip those too close to already-evaluated)
            for x in lattice:
                if not time_ok() or call_count >= budget:
                    break
                # Skip if very close to best_x
                if np.linalg.norm(x - best_x) < spacing * 0.1:
                    continue
                f = safe_call(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        # ---------- Phase 2: Coordinate-wise line search ----------
        # Golden-section search along each coordinate axis, iteratively
        x = best_x.copy()
        fx = best_f
        remaining = budget - call_count

        # Number of full coordinate sweeps
        max_sweeps = min(10, remaining // (dim * 10))  # each sweep uses ~dim*10 evals
        max_sweeps = max(max_sweeps, 2)

        for sweep in range(max_sweeps):
            if not time_ok() or call_count >= budget:
                break
            improved = False
            for d in range(dim):
                if not time_ok() or call_count >= budget:
                    break
                # Golden-section search along dimension d
                # Define interval [lo, hi] around current x[d]
                width = 2.0 / (sweep + 1)  # shrinking interval
                lo = x[d] - width
                hi = x[d] + width
                lo = max(lo, -5.0)
                hi = min(hi, 5.0)
                if hi - lo < 1e-6:
                    continue

                # Golden ratio
                gr = (math.sqrt(5.0) - 1.0) / 2.0
                x1 = hi - gr * (hi - lo)
                x2 = lo + gr * (hi - lo)

                # Evaluate x1 and x2
                x_test1 = x.copy()
                x_test1[d] = x1
                x_test2 = x.copy()
                x_test2[d] = x2
                f1 = safe_call(x_test1)
                f2 = safe_call(x_test2)

                # Also evaluate current point if not already known
                if not time_ok() or call_count >= budget:
                    break

                # Iterate golden-section for a few steps
                for _ in range(8):
                    if not time_ok() or call_count >= budget:
                        break
                    if f1 < f2:
                        hi = x2
                        x2 = x1
                        f2 = f1
                        x1 = hi - gr * (hi - lo)
                        x_test1 = x.copy()
                        x_test1[d] = x1
                        f1 = safe_call(x_test1)
                    else:
                        lo = x1
                        x1 = x2
                        f1 = f2
                        x2 = lo + gr * (hi - lo)
                        x_test2 = x.copy()
                        x_test2[d] = x2
                        f2 = safe_call(x_test2)

                # Best point along this axis
                if f1 < fx:
                    x[d] = x1
                    fx = f1
                    improved = True
                elif f2 < fx:
                    x[d] = x2
                    fx = f2
                    improved = True

            if not improved and sweep > 0:
                break

        return x, fx
# EVOLVE-BLOCK-END
