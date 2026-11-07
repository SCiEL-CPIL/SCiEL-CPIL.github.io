"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with LHS initialization and 1/5-rule adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # LHS-style diverse initial points for broad exploration
        def lhs_points(n, d, rng):
            pts = np.zeros((n, d))
            for dd in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    pts[perm[i], dd] = -5.0 + 10.0 * (i + 0.5) / n
            return pts

        best_x = None
        best_f = float('inf')
        call_count = 0

        # Phase 1: Diverse LHS sampling to find good starting regions
        n_lhs = min(30, budget // 10)
        lhs_pts = lhs_points(n_lhs, dim, rng)

        for pt in lhs_pts:
            if time.time() > deadline or call_count >= budget:
                break
            f_val = func(pt)
            call_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        # Phase 2: Multi-restart (1+1)-ES with 1/5-rule sigma adaptation
        remaining = budget - call_count
        if remaining <= 0:
            return best_x, best_f

        # Determine restarts and steps per restart
        # Cost per restart: 1 (initial eval) + per_restart steps
        # We want enough restarts for exploration but enough steps for exploitation
        per_restart = 35
        cost_per_restart = per_restart + 1
        n_restarts = max(1, remaining // cost_per_restart)

        # Build list of start points: best found + random diverse points
        start_points = [best_x.copy()]
        for _ in range(n_restarts - 1):
            start_points.append(rng.uniform(-5, 5, dim))

        for sp in start_points:
            if time.time() > deadline or call_count >= budget:
                break

            x = sp.copy()
            if call_count >= budget:
                break
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            # 1/5-rule adaptation: track successes in a window
            successes = 0
            window = 10

            for _ in range(per_restart):
                if time.time() > deadline or call_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    successes = max(0, successes - 1)

                # Adapt sigma every 10 evaluations
                if successes >= 5:
                    sigma *= 1.25
                    successes = 0
                elif successes <= 2:
                    sigma *= 0.8
                    successes = 0

                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
