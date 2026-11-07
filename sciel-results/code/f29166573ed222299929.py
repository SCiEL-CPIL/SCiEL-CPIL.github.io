"""Rewritten optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Coordinate descent with golden-section 1D refinement, interleaved with
    random restarts to escape local minima in the 21-peak landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def should_stop():
            return time.time() > deadline or calls >= budget

        def track(func_call, x):
            nonlocal calls, best_f, best_x
            fx = func_call(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(777)

        # Phase 1: Diverse initial exploration
        # Evaluate a set of structured points to get early progress
        # Center
        center = np.zeros(dim)
        track(func, center)

        # Axis-aligned points
        for dist in [1.0, 2.0, 3.0, 4.0]:
            for d in range(dim):
                if should_stop():
                    break
                for sign in [1, -1]:
                    if should_stop():
                        break
                    x = center.copy()
                    x[d] = sign * dist
                    track(func, x)

        # Some random points
        for _ in range(20):
            if should_stop():
                break
            x = rng.uniform(-4.5, 4.5, dim)
            track(func, x)

        if should_stop() or best_x is None:
            return best_x, best_f

        # Phase 2: Coordinate descent with golden-section refinement
        # Use the golden ratio for 1D search
        phi = (1 + np.sqrt(5)) / 2
        inv_phi = 1.0 / phi
        inv_phi2 = inv_phi * inv_phi

        def golden_section_1d(x, dim_idx, x_min, x_max, max_evals=10):
            """
            1D golden-section search along dim_idx of x.
            Returns (best_x_1d, best_f_1d, n_evals)
            """
            nonlocal calls, best_f, best_x

            a = x_min
            b = x_max
            c = b - inv_phi * (b - a)
            d = a + inv_phi * (b - a)

            # Evaluate c and d
            x_c = x.copy()
            x_c[dim_idx] = c
            x_d = x.copy()
            x_d[dim_idx] = d

            f_c = track(func, x_c)
            f_d = track(func, x_d)

            n_evals = 2

            for _ in range(max_evals - 2):
                if should_stop():
                    break
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - inv_phi * (b - a)
                    x_c = x.copy()
                    x_c[dim_idx] = c
                    f_c = track(func, x_c)
                    n_evals += 1
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + inv_phi * (b - a)
                    x_d = x.copy()
                    x_d[dim_idx] = d
                    f_d = track(func, x_d)
                    n_evals += 1

            # Return the better of c and d
            if f_c <= f_d:
                x_best = x_c
                f_best = f_c
            else:
                x_best = x_d
                f_best = f_d

            return x_best, f_best, n_evals

        # Coordinate descent loop
        # We'll do multiple rounds of coordinate descent, with occasional restarts
        x_current = best_x.copy()
        f_current = best_f

        # Number of coordinate descent rounds
        max_rounds = 30
        evals_per_round = max(5, (budget - calls) // (max_rounds * dim))

        for round_idx in range(max_rounds):
            if should_stop():
                break

            # Shuffle coordinate order for diversity
            coord_order = rng.permutation(dim)

            improved = False
            for dim_idx in coord_order:
                if should_stop():
                    break

                # Determine search range for this dimension
                # Use a range around the current value
                center_val = x_current[dim_idx]
                # Adaptive range: start wide, narrow as we converge
                range_width = max(1.0, 5.0 / (1 + round_idx * 0.5))
                x_min = max(-5.0, center_val - range_width)
                x_max = min(5.0, center_val + range_width)

                if x_max - x_min < 1e-6:
                    continue

                # Perform golden-section search
                x_new, f_new, n_evals = golden_section_1d(
                    x_current, dim_idx, x_min, x_max,
                    max_evals=min(evals_per_round, 12)
                )

                if f_new < f_current - 1e-10:
                    x_current = x_new
                    f_current = f_new
                    improved = True

            # If no improvement in this round, try a restart
            if not improved:
                # Random restart: perturb current best
                if should_stop():
                    break
                n_restarts = 5
                for _ in range(n_restarts):
                    if should_stop():
                        break
                    # Perturb best_x
                    x_restart = best_x + 2.0 * rng.standard_normal(dim)
                    x_restart = np.clip(x_restart, -5, 5)
                    f_restart = track(func, x_restart)

                    # Quick coordinate descent from restart point
                    x_c = x_restart.copy()
                    f_c = f_restart
                    for dim_idx in rng.permutation(dim):
                        if should_stop():
                            break
                        center_val = x_c[dim_idx]
                        range_width = 2.0
                        x_min = max(-5.0, center_val - range_width)
                        x_max = min(5.0, center_val + range_width)
                        if x_max - x_min < 1e-6:
                            continue
                        x_new, f_new, _ = golden_section_1d(
                            x_c, dim_idx, x_min, x_max, max_evals=6
                        )
                        if f_new < f_c:
                            x_c = x_new
                            f_c = f_new

                    if f_c < best_f:
                        best_x = x_c
                        best_f = f_c
                        x_current = x_c.copy()
                        f_current = f_c
                        break  # Restart worked, continue with this point

        # Phase 3: Final refinement with tighter golden-section searches
        if not should_stop() and best_x is not None:
            # Do a few more rounds with tighter ranges
            for _ in range(5):
                if should_stop():
                    break
                coord_order = rng.permutation(dim)
                for dim_idx in coord_order:
                    if should_stop():
                        break
                    center_val = best_x[dim_idx]
                    range_width = 0.5
                    x_min = max(-5.0, center_val - range_width)
                    x_max = min(5.0, center_val + range_width)
                    if x_max - x_min < 1e-6:
                        continue
                    x_new, f_new, _ = golden_section_1d(
                        best_x, dim_idx, x_min, x_max, max_evals=8
                    )
                    if f_new < best_f:
                        best_x = x_new
                        best_f = f_new

        return best_x, best_f
# EVOLVE-BLOCK-END
