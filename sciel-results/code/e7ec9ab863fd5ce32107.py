import numpy as np
import time

class Optimizer:
    """
    Optimizer using Adaptive Coordinate Descent with Nelder-Mead refinement
    for BBOB f13 (Sharp Ridge).
    
    Combines the structural idea of coordinate descent (exploiting anisotropic
    ridge geometry with per-axis adaptive steps) with the refinement machinery
    of Nelder-Mead (robust local convergence).
    
    Strategy:
    1. Global exploration with random sampling to find the basin.
    2. Adaptive coordinate descent: iterate over dimensions, doing 1D line
       search along each axis with step sizes that grow/shrink based on progress.
       This exploits the narrow/wide structure of the sharp ridge.
    3. Periodic Nelder-Mead refinement from the best point found by CD to
       polish the solution and handle any remaining non-axis-aligned structure.
    4. Restarts when progress stalls.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin

        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val is None or np.isnan(f_val) or np.isinf(f_val):
                    return None
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration
        n_random = 20
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0.copy()
            best_f = f0

        # Nelder-Mead implementation for refinement
        def nelder_mead(x0, f0, max_iter=30):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return

            # Initialize simplex
            simplex = [x0.copy()]
            f_values = [f0]

            # Create other vertices by perturbing each coordinate
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += 0.05
                simplex.append(x_new)
                f_val = safe_call(x_new)
                if f_val is None:
                    return
                f_values.append(f_val)

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for iteration in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    return

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check for convergence
                if abs(f_values[0] - f_values[-1]) < 1e-12:
                    break

                # Compute centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = safe_call(x_r)
                if f_r is None:
                    return

                if f_values[0] <= f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = safe_call(x_e)
                    if f_e is None:
                        return
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    if f_r < f_values[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = safe_call(x_c)
                    if f_c is None:
                        return

                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_val = safe_call(simplex[i])
                            if f_val is None:
                                return
                            f_values[i] = f_val

                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()

        # Adaptive Coordinate Descent
        def coord_descent(x_start, max_rounds=50):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return

            x = x_start.copy()
            f_curr = safe_call(x)
            if f_curr is None:
                return

            # Per-axis step sizes, initialized adaptively
            steps = np.ones(dim) * 0.5
            min_step = 1e-6
            max_step = 10.0

            for rnd in range(max_rounds):
                if calls >= budget or time.time() > deadline:
                    return

                improved = False
                for i in range(dim):
                    if calls >= budget or time.time() > deadline:
                        return

                    # 1D line search along axis i
                    # Try positive and negative directions
                    for direction in [1, -1]:
                        step = steps[i]
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        f_trial = safe_call(x_trial)
                        if f_trial is None:
                            return

                        if f_trial < f_curr:
                            # Move in this direction
                            x = x_trial
                            f_curr = f_trial
                            improved = True
                            # Grow step
                            steps[i] = min(steps[i] * 1.5, max_step)
                            # Continue searching in same direction
                            while calls < budget and time.time() < deadline:
                                x_trial = x.copy()
                                x_trial[i] += direction * step
                                x_trial = np.clip(x_trial, -5.0, 5.0)
                                f_trial = safe_call(x_trial)
                                if f_trial is None:
                                    break
                                if f_trial < f_curr:
                                    x = x_trial
                                    f_curr = f_trial
                                    improved = True
                                    steps[i] = min(steps[i] * 1.3, max_step)
                                else:
                                    break
                        else:
                            # Shrink step for this axis
                            steps[i] = max(steps[i] * 0.5, min_step)

                    if steps[i] < min_step * 2:
                        # Axis is essentially converged, skip
                        continue

                # Update global best
                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x.copy()

                if not improved:
                    break

            # Final polish with Nelder-Mead from the CD result
            if calls < budget - 10 and time.time() < deadline:
                nelder_mead(x, f_curr, max_iter=15)

        # Main loop: Alternate between CD and NM with restarts
        f_prev = best_f
        restart_count = 0
        max_restarts = 10

        while calls < budget and time.time() < deadline:
            if calls >= budget:
                break

            # Run coordinate descent from current best
            coord_descent(best_x, max_rounds=30)

            # Check if we made progress
            if best_f < f_prev - 1e-8:
                f_prev = best_f
                restart_count = 0
            else:
                restart_count += 1

            # If no progress, try a restart
            if restart_count >= 2:
                if calls < budget - 20 and restart_count <= max_restarts:
                    # Random restart
                    x_new = rng.uniform(-5.0, 5.0, size=dim)
                    f_new = safe_call(x_new)
                    if f_new is None:
                        break
                    if f_new < best_f:
                        best_x = x_new
                        best_f = f_new
                        f_prev = best_f
                    else:
                        # Try NM from the new point if it's not too bad
                        if f_new < best_f + 50:
                            nelder_mead(x_new, f_new, max_iter=10)
                        restart_count = 0
                else:
                    break

            # If we're close to budget, do a final NM polish
            if budget - calls < 15:
                nelder_mead(best_x, best_f, max_iter=5)
                break

        return best_x, best_f
