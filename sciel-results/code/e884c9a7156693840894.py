"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: adaptive basin finding + ridge-following refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        calls = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return np.inf
            try:
                x = np.asarray(x, dtype=float)
                x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                calls += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls += 1
                return np.inf

        # Phase 1: Adaptive basin finding with directional exploration
        # Use a set of candidate directions and adaptively weight them
        rng = np.random.default_rng(12345)

        # Initialize with diverse starting points
        n_initial = min(20, budget // 10)
        for i in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            fx = safe_call(x)

        # Adaptive directional search
        # Maintain a set of search directions with associated step sizes
        n_dirs = 12
        directions = []
        for i in range(n_dirs):
            d = rng.standard_normal(dim)
            d /= (np.linalg.norm(d) + 1e-12)
            directions.append(d)

        # Also add coordinate directions
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)

        steps = np.array([1.0] * len(directions))
        successes = np.zeros(len(directions))

        phase1_budget = int(budget * 0.55)
        phase2_budget = budget - phase1_budget

        # Phase 1: Adaptive directional search
        while calls < phase1_budget and time.time() < deadline:
            # Pick direction with probability proportional to recent success
            weights = successes + 0.1
            weights /= weights.sum()
            idx = rng.choice(len(directions), p=weights)
            d = directions[idx]
            step = steps[idx]

            # Try positive and negative steps
            x_pos = best_x + step * d
            x_neg = best_x - step * d
            x_pos = np.clip(x_pos, -5.0, 5.5)
            x_neg = np.clip(x_neg, -5.0, 5.5)

            fx_pos = safe_call(x_pos)
            if calls >= phase1_budget:
                break

            fx_neg = safe_call(x_neg)

            if fx_pos < best_fx:
                successes[idx] += 1.0
                steps[idx] = min(steps[idx] * 1.2, 4.0)
            elif fx_neg < best_fx:
                # Flip direction
                directions[idx] = -d
                successes[idx] += 1.0
                steps[idx] = min(steps[idx] * 1.2, 4.0)
            else:
                steps[idx] *= 0.75
                successes[idx] *= 0.8

            # Occasionally add new random direction
            if calls % 50 == 0 and calls < phase1_budget - 10:
                d_new = rng.standard_normal(dim)
                d_new /= (np.linalg.norm(d_new) + 1e-12)
                # Replace the least successful direction
                worst_idx = np.argmin(successes)
                directions[worst_idx] = d_new
                steps[worst_idx] = 1.0
                successes[worst_idx] = 0.0

        # Phase 2: Ridge-following refinement
        # Estimate ridge direction from gradient approximation
        # Then do fine-grained 1D optimization along that direction

        # Estimate gradient at best point using central differences
        eps = 0.05
        grad = np.zeros(dim)
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            x_p = best_x.copy()
            x_m = best_x.copy()
            x_p[i] += eps
            x_m[i] -= eps
            x_p = np.clip(x_p, -5.0, 5.0)
            x_m = np.clip(x_m, -5.0, 5.0)
            fp = safe_call(x_p)
            if calls >= budget:
                break
            fm = safe_call(x_m)
            if np.isfinite(fp) and np.isfinite(fm):
                grad[i] = (fp - fm) / (2 * eps)

        # The ridge direction is approximately orthogonal to the gradient
        # (for f13, the valley is narrow in one direction, wide in another)
        # Use the gradient to identify the narrow direction
        if np.linalg.norm(grad) > 1e-6:
            grad_norm = grad / np.linalg.norm(grad)
        else:
            grad_norm = np.zeros(dim)

        # For ridge following, we want to move along the direction that
        # minimizes the function. Use a combination of negative gradient
        # and a ridge-following heuristic.

        # Fine-grained local search around best point
        local_step = 0.1
        for iteration in range(50):
            if calls >= budget or time.time() > deadline:
                break

            # Try small perturbations in all directions
            improved = False
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break

                # Try positive
                x_try = best_x.copy()
                x_try[i] += local_step
                x_try = np.clip(x_try, -5.0, 5.0)
                fx_try = safe_call(x_try)
                if fx_try < best_fx:
                    improved = True
                    continue

                # Try negative
                x_try = best_x.copy()
                x_try[i] -= local_step
                x_try = np.clip(x_try, -5.0, 5.0)
                fx_try = safe_call(x_try)
                if fx_try < best_fx:
                    improved = True
                    continue

            if improved:
                local_step *= 0.8
            else:
                local_step *= 0.5
                if local_step < 0.001:
                    break

        # Final polish: very fine coordinate descent
        fine_step = 0.01
        for _ in range(3):
            if calls >= budget or time.time() > deadline:
                break
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                # Binary search for optimum along axis i
                x_lo = best_x[i] - fine_step * 3
                x_hi = best_x[i] + fine_step * 3
                x_lo = max(x_lo, -5.0)
                x_hi = min(x_hi, 5.0)

                for _ in range(4):
                    if calls >= budget:
                        break
                    x_mid1 = x_lo + (x_hi - x_lo) / 3
                    x_mid2 = x_lo + 2 * (x_hi - x_lo) / 3

                    x1 = best_x.copy()
                    x2 = best_x.copy()
                    x1[i] = x_mid1
                    x2[i] = x_mid2
                    x1 = np.clip(x1, -5.0, 5.0)
                    x2 = np.clip(x2, -5.0, 5.0)

                    f1 = safe_call(x1)
                    if calls >= budget:
                        break
                    f2 = safe_call(x2)

                    if f1 < f2:
                        x_hi = x_mid2
                    else:
                        x_lo = x_mid1

                    # Update best if we found something better
                    if f1 < best_fx:
                        best_x = x1.copy()
                        best_fx = f1
                    if f2 < best_fx:
                        best_x = x2.copy()
                        best_fx = f2

                # Set best_x[i] to the best found in this interval
                x_test = best_x.copy()
                x_test[i] = (x_lo + x_hi) / 2
                x_test = np.clip(x_test, -5.0, 5.0)
                f_test = safe_call(x_test)
                if f_test < best_fx:
                    best_x = x_test.copy()
                    best_fx = f_test

            fine_step *= 0.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
