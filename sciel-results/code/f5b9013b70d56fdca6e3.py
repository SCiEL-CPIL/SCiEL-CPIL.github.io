"""Optimizer for BBOB f13 (Sharp ridge) with early-progress-focused design."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate-wise golden-section search with structured initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s kill

        # --- Structured initialization: probe center and coordinate extremes ---
        # This gives immediate information about the landscape shape
        x_center = np.zeros(dim)
        f_center = func(x_center)
        calls = 1

        # Probe each coordinate at a mid-range offset to get direction info
        # Use a moderate offset (2.5) to stay well within bounds
        offset = 2.5
        x_plus = x_center.copy()
        x_minus = x_center.copy()
        for i in range(dim):
            x_plus[i] = offset
            x_minus[i] = -offset
            fp = func(x_plus)
            fm = func(x_minus)
            calls += 2
            if fp < f_center:
                x_center = x_plus.copy()
                f_center = fp
            if fm < f_center:
                x_center = x_minus.copy()
                f_center = fm
            x_plus[i] = 0.0
            x_minus[i] = 0.0

        # Now we have a good starting point. Do coordinate-wise golden-section
        # search to refine each dimension.
        # We'll do multiple passes over coordinates, each pass narrowing the range.

        # Initialize bounds around the best point found
        best_x = x_center.copy()
        best_f = f_center

        # For each coordinate, we'll do a 1D golden-section search
        # within a local window. We do several rounds with shrinking windows.

        # Golden ratio
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618

        # Number of 1D evaluations per coordinate per pass
        # We have budget - calls remaining. Let's allocate carefully.
        remaining = budget - calls

        # We'll do passes. In each pass, for each coordinate, do a small 1D search.
        # Let's figure out how many evaluations per coordinate per pass.
        # Total evals per pass = dim * evals_per_coord
        # We want to use up the budget.

        # Strategy: do a fixed number of 1D golden-section iterations per coordinate.
        # Each coordinate search uses 2 evals per iteration (after initial setup).
        # Let's do it adaptively: keep going until budget or time runs out.

        # For each coordinate, maintain a [lo, hi] interval and do golden-section.
        # Initialize intervals to [-5, 5] for each coordinate.

        lo = np.full(dim, -5.0)
        hi = np.full(dim, 5.0)

        # We'll iterate: pick a coordinate, do one golden-section step on it.
        # Cycle through coordinates round-robin.

        # For each coordinate, we need two interior points a and b.
        # a = lo + gr*(hi - lo), b = hi - gr*(hi - lo)
        # Evaluate f at a and b (with other coords fixed at best_x).
        # Keep the better one, discard the worse, shrink interval.

        # State per coordinate: interval [lo_i, hi_i], and the two interior points
        # and their function values.

        # Let's precompute the two interior point positions for each coordinate.
        # For coordinate i:
        #   a_i = lo_i + gr*(hi_i - lo_i)
        #   b_i = hi_i - gr*(hi_i - lo_i)
        # We evaluate x with coord i set to a_i, and x with coord i set to b_i.

        # We'll maintain:
        #   x_a[i] = best_x with coord i = a_i
        #   f_a[i] = func(x_a[i])
        #   x_b[i] = best_x with coord i = b_i
        #   f_b[i] = func(x_b[i])

        # But best_x changes as we improve, so this gets complex.
        # Simpler: for each coordinate, do a full 1D search in a local window
        # centered at the current best_x[i], with a fixed window size that shrinks.

        # Let me restructure: do rounds. In each round, for each coordinate,
        # do a small 1D golden-section search in a window around best_x[i].

        # Round 1: window = 4.0 (search in [best_x[i]-4, best_x[i]+4] clipped to [-5,5])
        # Round 2: window = 1.0
        # Round 3: window = 0.3
        # Round 4: window = 0.1
        # etc.

        # Each 1D search uses a fixed number of evaluations, say 8.
        # Total per round = dim * 8 = 40.
        # We have ~450 remaining, so ~11 rounds.

        # But we should stop early if no improvement.

        # Let's just do a loop: while budget allows and time allows,
        # cycle through coordinates and do one golden-section step each.

        # I'll go with the round-robin golden-section approach.

        # Initialize per-coordinate state
        # For each coord i, we have interval [lo[i], hi[i]] and two interior points.
        # We need to evaluate both interior points first (2 evals per coord = 10 total).
        # Then each subsequent step uses 1 eval per coord.

        # Let's set up:
        # For coord i, interior points:
        #   a_i = lo[i] + gr*(hi[i] - lo[i])
        #   b_i = hi[i] - gr*(hi[i] - lo[i])

        # Evaluate all a_i and b_i points.
        f_a = np.zeros(dim)
        f_b = np.zeros(dim)
        a_pos = np.zeros(dim)
        b_pos = np.zeros(dim)

        for i in range(dim):
            a_pos[i] = lo[i] + gr * (hi[i] - lo[i])
            b_pos[i] = hi[i] - gr * (hi[i] - lo[i])

            x_a = best_x.copy()
            x_a[i] = a_pos[i]
            f_a[i] = func(x_a)
            calls += 1

            x_b = best_x.copy()
            x_b[i] = b_pos[i]
            f_b[i] = func(x_b)
            calls += 1

            if f_a[i] < best_f:
                best_f = f_a[i]
                best_x[i] = a_pos[i]
            if f_b[i] < best_f:
                best_f = f_b[i]
                best_x[i] = b_pos[i]

        # Now do round-robin golden-section steps.
        # For each coord i, compare f_a[i] and f_b[i]:
        #   If f_a < f_b: hi[i] = b_pos[i], b_pos[i] = a_pos[i], f_b[i] = f_a[i],
        #                  a_pos[i] = lo[i] + gr*(hi[i]-lo[i]), evaluate new a
        #   If f_b < f_a: lo[i] = a_pos[i], a_pos[i] = b_pos[i], f_a[i] = f_b[i],
        #                  b_pos[i] = hi[i] - gr*(hi[i]-lo[i]), evaluate new b
        #   If equal: shrink both (rare)

        # We track which coord to evaluate next.
        # We'll cycle through coords. For each, do one step (1 eval).

        # But we need to be careful: when we update best_x, the other coordinates'
        # interior points are no longer valid because they were evaluated with
        # the old best_x for the other coords. This is a problem.

        # Simpler approach: do full 1D searches sequentially, one coordinate at a time,
        # updating best_x as we go. Each 1D search uses the current best_x for
        # other coordinates.

        # Let me restart with a cleaner approach:
        # Sequential coordinate-wise 1D golden-section search, multiple passes.

        # Actually, let me just do a simple but effective approach:
        # 1. Already have a good starting point from the initialization phase.
        # 2. Do coordinate-wise line searches: for each coordinate, do a 1D
        #    golden-section search in a window, update best_x.
        # 3. Repeat with smaller windows.
        # 4. When budget is nearly exhausted, do fine random perturbations.

        # Let me implement this cleanly.

        # Reset: we've used `calls` evaluations. Let's do the rest.

        # Function to do a 1D golden-section search on coordinate i
        # within [center - half, center + half], clipped to [-5, 5].
        # Uses `n_evals` function evaluations.
        # Returns the best point found and its value.

        def golden_1d(func, best_x, i, center, half, n_evals, calls_ref):
            """Do 1D golden-section search on coordinate i."""
            lo_i = max(-5.0, center - half)
            hi_i = min(5.0, center + half)
            if hi_i - lo_i < 1e-10:
                return center, func(best_x), calls_ref[0]

            # Two interior points
            a = lo_i + gr * (hi_i - lo_i)
            b = hi_i - gr * (hi_i - lo_i)

            x_a = best_x.copy()
            x_a[i] = a
            f_a = func(x_a)
            calls_ref[0] += 1

            x_b = best_x.copy()
            x_b[i] = b
            f_b = func(x_b)
            calls_ref[0] += 1

            # Track best
            if f_a < f_b:
                best_1d = a
                best_f_1d = f_a
            else:
                best_1d = b
                best_f_1d = f_b

            # Iteratively shrink
            for _ in range(n_evals - 2):
                if calls_ref[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                if f_a < f_b:
                    hi_i = b
                    b = a
                    f_b = f_a
                    a = lo_i + gr * (hi_i - lo_i)
                    x_a = best_x.copy()
                    x_a[i] = a
                    f_a = func(x_a)
                    calls_ref[0] += 1
                    if f_a < best_f_1d:
                        best_f_1d = f_a
                        best_1d = a
                else:
                    lo_i = a
                    a = b
                    f_a = f_b
                    b = hi_i - gr * (hi_i - lo_i)
                    x_b = best_x.copy()
                    x_b[i] = b
                    f_b = func(x_b)
                    calls_ref[0] += 1
                    if f_b < best_f_1d:
                        best_f_1d = f_b
                        best_1d = b

            return best_1d, best_f_1d, calls_ref[0]

        # Now do multiple passes with shrinking windows
        # Pass 1: half = 4.0, n_evals per coord = 10
        # Pass 2: half = 1.0, n_evals per coord = 10
        # Pass 3: half = 0.3, n_evals per coord = 8
        # Pass 4: half = 0.1, n_evals per coord = 6
        # Pass 5: half = 0.03, n_evals per coord = 4
        # etc.

        calls_ref = [calls]
        best_x = best_x.copy()
        best_f = best_f

        passes = [
            (4.0, 10),
            (1.0, 10),
            (0.3, 8),
            (0.1, 6),
            (0.03, 5),
            (0.01, 4),
            (0.003, 3),
            (0.001, 3),
        ]

        for half, n_evals in passes:
            if calls_ref[0] >= budget - 1:
                break
            if time.time() > deadline:
                break
            for i in range(dim):
                if calls_ref[0] >= budget - 1:
                    break
                if time.time() > deadline:
                    break
                # Do 1D search on coord i
                new_val, new_f, calls_ref[0] = golden_1d(
                    func, best_x, i, best_x[i], half, n_evals, calls_ref
                )
                if new_f < best_f:
                    best_f = new_f
                    best_x[i] = new_val

        # Use remaining budget for fine random perturbations around best_x
        if calls_ref[0] < budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            sigma = 0.01
            while calls_ref[0] < budget and time.time() < deadline:
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_ref[0] += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
