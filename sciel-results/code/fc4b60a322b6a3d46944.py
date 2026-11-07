"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with multiple restarts and adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 30s safety margin before 330s kill

        # Count function calls
        calls = [0]

        def counted_func(x):
            calls[0] += 1
            if calls[0] > self.budget:
                # Should not happen, but guard against it
                return 1e10
            return float(func(np.clip(x, -5, 5)))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def nelder_mead(x0, sigma_init, max_iter):
            """Run Nelder-Mead from x0 with initial simplex size sigma_init."""
            dim = self.dim
            # Create initial simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += sigma_init

            # Evaluate initial simplex
            fvals = np.array([counted_func(v) for v in simplex])

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if calls[0] >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = counted_func(x_r)

                if calls[0] >= self.budget:
                    break
                if time.time() > deadline:
                    break

                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = counted_func(x_e)

                    if calls[0] >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Reflection is good but not better than best
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)

                    x_c = np.clip(x_c, -5, 5)
                    f_c = counted_func(x_c)

                    if calls[0] >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = counted_func(simplex[i])

                            if calls[0] >= self.budget:
                                break
                            if time.time() > deadline:
                                return

                        if calls[0] >= self.budget or time.time() > deadline:
                            break

                # Update best
                best_idx = np.argmin(fvals)
                if fvals[best_idx] < best_f[0]:
                    best_f[0] = fvals[best_idx]
                    best_x[0] = simplex[best_idx].copy()

            # Return best from this run
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Use a mutable container for best
        best_f = [np.inf]
        best_x = [None]

        # Multiple restarts with different initializations
        # Strategy: start with a few random points, then refine
        n_restarts = 8

        # First, do a quick random search to find a good starting point
        n_random = min(20, self.budget // 5)
        for i in range(n_random):
            if calls[0] >= self.budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-4, 4, self.dim)
            f = counted_func(x)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x.copy()

        # Now do Nelder-Mead from the best point found
        if best_x[0] is not None:
            # Start with a moderate step size
            sigma_init = 1.0
            max_iter = (self.budget - calls[0]) // (n_restarts + 1)
            max_iter = max(max_iter, 10)

            x_start = best_x[0].copy()
            for restart in range(n_restarts):
                if calls[0] >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # For subsequent restarts, use smaller step sizes
                if restart == 0:
                    x_curr = x_start
                    sig = sigma_init
                else:
                    # Perturb the best found so far
                    x_curr = best_x[0] + rng.normal(0, 0.5 * sigma_init / (restart + 1), self.dim)
                    x_curr = np.clip(x_curr, -4, 4)
                    sig = sigma_init / (restart + 1)

                x_res, f_res = nelder_mead(x_curr, sig, max_iter)
                if f_res < best_f[0]:
                    best_f[0] = f_res
                    best_x[0] = x_res.copy()

        # If we have budget left, do one more fine-tuning Nelder-Mead
        if best_x[0] is not None and calls[0] < self.budget and time.time() < deadline:
            remaining = self.budget - calls[0]
            if remaining > 10:
                x_res, f_res = nelder_mead(best_x[0], 0.1, remaining - 2)
                if f_res < best_f[0]:
                    best_f[0] = f_res
                    best_x[0] = x_res.copy()

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
