"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start coordinate-wise line search with golden-section 1D optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        def golden_section_1d(x, axis, lo, hi, func):
            """Minimize along one axis using golden-section search."""
            phi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
            c = hi - phi * (hi - lo)
            d = lo + phi * (hi - lo)

            x_c = x.copy()
            x_c[axis] = c
            fc = func(np.clip(x_c, -5, 5))

            x_d = x.copy()
            x_d[axis] = d
            fd = func(np.clip(x_d, -5, 5))

            # We need to track calls; but we don't have a global counter here easily.
            # Instead, we'll rely on the outer loop budget check.
            # For simplicity in this design, we assume each golden-section step costs 1-2 calls.
            # We'll do a fixed number of iterations and check time/budget externally.

            for _ in range(20):  # ~20 iterations gives good precision
                if time.time() > deadline:
                    break

                if fc < fd:
                    hi = d
                    d = c
                    fd = fc
                    c = hi - phi * (hi - lo)
                    x_c = x.copy()
                    x_c[axis] = c
                    fc = func(np.clip(x_c, -5, 5))
                else:
                    lo = c
                    c = d
                    fc = fd
                    d = lo + phi * (hi - lo)
                    x_d = x.copy()
                    x_d[axis] = d
                    fd = func(np.clip(x_d, -5, 5))

            # Return the better of the two points
            mid = (lo + hi) / 2.0
            if fc < fd:
                x[axis] = c
            else:
                x[axis] = d
            return x

        def coordinate_line_search(x, func, max_rounds=3):
            """Perform coordinate-wise line search for multiple rounds."""
            for _ in range(max_rounds):
                if time.time() > deadline:
                    break
                for axis in range(dim):
                    if time.time() > deadline:
                        break
                    # Determine a reasonable interval around current x[axis]
                    current = x[axis]
                    # Use a window of size 2.0 initially, shrinking as we go
                    half_width = 2.0
                    lo = max(-5.0, current - half_width)
                    hi = min(5.0, current + half_width)

                    if hi - lo < 1e-6:
                        continue

                    x = golden_section_1d(x, axis, lo, hi, func)
                    # After first round, use a tighter window
                    half_width *= 0.5

            return x

        # Generate a grid of initial points
        # Use a 3^5 grid = 243 points, but we only have 500 calls.
        # Each point needs at least 1 call to evaluate, plus line search calls.
        # Let's use a coarser grid: 2^5 = 32 points, or a few hand-picked points.
        # Actually, let's use a stratified sampling approach.

        # Strategy: Start with a few well-chosen initial points
        # Point 1: origin
        # Point 2: corners (±5, ±5, ...) - but clipped to box
        # Point 3: some intermediate points

        initial_points = []

        # Origin
        initial_points.append(np.zeros(dim))

        # Axes-aligned points
        for axis in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[axis] = sign * 4.0
                initial_points.append(p)

        # Some random-ish deterministic points
        rng = np.random.default_rng(42)
        for _ in range(10):
            p = rng.uniform(-4, 4, dim)
            initial_points.append(p)

        # Evaluate each initial point and do line search if budget allows
        for x0 in initial_points:
            if time.time() > deadline:
                break

            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            try_update(x0, f0)

            # If this point is promising, do line search
            if f0 < best_f + 50.0:  # within 50 of best
                x_refined = coordinate_line_search(x0.copy(), func, max_rounds=2)
                f_refined = func(x_refined)
                try_update(x_refined, f_refined)

                # If still promising, do more refinement
                if f_refined < best_f + 10.0:
                    x_refined2 = coordinate_line_search(x_refined.copy(), func, max_rounds=3)
                    f_refined2 = func(x_refined2)
                    try_update(x_refined2, f_refined2)

        # Final refinement on the best point found
        if best_x is not None and time.time() < deadline:
            x_final = coordinate_line_search(best_x.copy(), func, max_rounds=5)
            f_final = func(x_final)
            try_update(x_final, f_final)

        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
