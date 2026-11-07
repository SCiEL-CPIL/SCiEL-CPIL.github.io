import numpy as np
import time

class Optimizer:
    """Coordinate-wise sequential search exploiting Bent Cigar structure."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            if time.time() >= deadline:
                return None
            val = func(np.asarray(x))
            calls += 1
            return val

        # Initial point: center of box
        x = np.zeros(dim)
        fx = safe_call(x)
        if fx is None:
            return x, np.inf
        best_x = x.copy()
        best_fx = fx

        # Phase 1: Coordinate-wise golden-section search
        # For each dimension, do a 1D optimization holding others fixed
        # Cycle through dimensions multiple times with shrinking ranges

        # Start with full range [-5, 5] for each dimension
        # Use golden-section ratio
        gr = (np.sqrt(5) - 1) / 2  # ~0.618
        inv_gr = 1.0 / gr  # ~1.618

        # For each dimension, we'll do several rounds of line search
        # Each round uses a few function evaluations to narrow the interval

        # We have budget - 1 calls remaining
        remaining = budget - 1

        # Strategy: For each coordinate, do a coarse-to-fine search
        # Divide remaining budget among dimensions and iterations

        # Let's do multiple passes through all dimensions
        # Each pass, we refine the estimate of the optimum for each coordinate

        # Number of passes: let's use 5 passes
        # Each pass, for each dimension, we do a ternary/golden search with limited evals

        # Actually, let's think differently. The Bent Cigar has one very flat direction
        # and others that are steep. A coordinate search should quickly find the steep
        # directions and then spend time on the flat one.

        # Let's do adaptive allocation: for each dimension, estimate curvature
        # by doing a few probes, then allocate more budget to flatter directions.

        # Simpler approach: cycle through dimensions, doing 2-3 golden-section
        # evaluations per dimension per cycle, for many cycles.

        # Let's estimate: with 499 remaining calls, and 5 dimensions,
        # if we do 3 evals per dimension per cycle, that's 15 per cycle,
        # giving ~33 cycles.

        # But we can be smarter: use the current best point and do local
        # line searches with shrinking step sizes.

        # Approach: For each dimension i, maintain a local interval [lo_i, hi_i]
        # Initially [-5, 5]. Each cycle, for each dimension, do a golden-section
        # step to narrow the interval.

        # Let's implement: for each dimension, keep track of the best value found
        # and a local search interval around it.

        # Actually, the simplest effective approach:
        # 1. For each dimension, do a coarse scan to find roughly where the minimum is
        # 2. Then do fine golden-section searches on each dimension

        # Coarse scan: for each dimension, evaluate at 5 points in [-5,5]
        # That's 5*5 = 25 calls for coarse scan

        # Then for each dimension, do golden-section with ~90 calls
        # 5 * 90 = 450 calls for fine search
        # Total: 25 + 450 = 475, plus initial = 476, within budget

        # Let's do:
        # Phase 1: Coarse scan - 5 points per dimension = 25 calls
        # Phase 2: Fine golden-section per dimension - ~90 calls each = 450 calls

        # Phase 1: Coarse scan
        scan_points = np.linspace(-5, 5, 5)
        for i in range(dim):
            for sp in scan_points:
                trial = best_x.copy()
                trial[i] = sp
                val = safe_call(trial)
                if val is not None and val < best_fx:
                    best_fx = val
                    best_x = trial.copy()

        if time.time() >= deadline or calls >= budget:
            return best_x, best_fx

        # Phase 2: Golden-section search per dimension
        # For each dimension, do a 1D golden-section search
        # We'll do multiple iterations per dimension

        # Remaining budget
        remaining = budget - calls

        # Allocate equally among dimensions
        per_dim = remaining // dim
        if per_dim < 5:
            per_dim = max(5, remaining // dim)

        for i in range(dim):
            if calls >= budget or time.time() >= deadline:
                break

            # For dimension i, do golden-section search
            # Start with interval around current best
            lo = max(-5.0, best_x[i] - 5.0)
            hi = min(5.0, best_x[i] + 5.0)

            # Golden-section search
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)

            # Evaluate at x1 and x2
            trial1 = best_x.copy()
            trial1[i] = x1
            val1 = safe_call(trial1)
            if val1 is None:
                break

            trial2 = best_x.copy()
            trial2[i] = x2
            val2 = safe_call(trial2)
            if val2 is None:
                break

            # Update best if improved
            if val1 < best_fx:
                best_fx = val1
                best_x = trial1.copy()
            if val2 < best_fx:
                best_fx = val2
                best_x = trial2.copy()

            # Continue golden-section iterations
            for _ in range(per_dim - 2):
                if calls >= budget or time.time() >= deadline:
                    break

                if val1 < val2:
                    # Minimum is in [lo, x2]
                    hi = x2
                    x2 = x1
                    val2 = val1
                    x1 = hi - gr * (hi - lo)
                else:
                    # Minimum is in [x1, hi]
                    lo = x1
                    x1 = x2
                    val1 = val2
                    x2 = lo + gr * (hi - lo)

                trial = best_x.copy()
                trial[i] = x1
                val1 = safe_call(trial)
                if val1 is None:
                    break
                if val1 < best_fx:
                    best_fx = val1
                    best_x = trial.copy()

                if calls >= budget or time.time() >= deadline:
                    break

                trial = best_x.copy()
                trial[i] = x2
                val2 = safe_call(trial)
                if val2 is None:
                    break
                if val2 < best_fx:
                    best_fx = val2
                    best_x = trial.copy()

        return best_x, best_fx
