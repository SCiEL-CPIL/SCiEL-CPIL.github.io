"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with adaptive reset and multi-start diversification."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Counter for function calls
        calls = [0]
        max_calls = budget

        def counted_func(x):
            if calls[0] >= max_calls:
                return float('inf')
            calls[0] += 1
            x = np.clip(x, -5, 5)
            return func(x)

        def nelder_mead(x0, fx0, remaining_budget, sigma=1.0):
            """Run Nelder-Mead simplex optimization."""
            dim = self.dim
            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1, i] += sigma
            fvals = np.zeros(dim + 1)
            fvals[0] = fx0
            for i in range(dim):
                fvals[i + 1] = counted_func(simplex[i + 1])

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_shrink = 0.5  # shrink

            # Track best
            best_idx = np.argmin(fvals)
            best_x = simplex[best_idx].copy()
            best_f = fvals[best_idx]

            iters = 0
            max_iters = 2000

            while calls[0] < max_calls and iters < max_iters:
                if time.time() > deadline:
                    break
                iters += 1

                # Sort simplex
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = counted_func(x_ref)

                if fvals[0] <= f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                elif f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = counted_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    f_con = counted_func(x_con)
                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fvals[i] = counted_func(simplex[i])

                # Update best
                best_idx = np.argmin(fvals)
                if fvals[best_idx] < best_f:
                    best_f = fvals[best_idx]
                    best_x = simplex[best_idx].copy()

            return best_x, best_f

        # Multi-start strategy with Nelder-Mead
        rng = np.random.default_rng(42)
        best_x = None
        best_f = float('inf')

        # First start: center
        x0 = np.zeros(dim)
        fx0 = counted_func(x0)
        best_x, best_f = nelder_mead(x0, fx0, budget)

        # Additional starts if budget remains
        num_extra_starts = 5
        for start_i in range(num_extra_starts):
            if calls[0] >= max_calls:
                break
            if time.time() > deadline - 10:
                break

            # Diversified starting points
            if start_i == 0:
                x0 = rng.uniform(-3, 3, dim)
            elif start_i == 1:
                x0 = rng.uniform(-5, -1, dim)
            elif start_i == 2:
                x0 = rng.uniform(1, 5, dim)
            elif start_i == 3:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Perturb the current best
                x0 = best_x + rng.standard_normal(dim) * 0.5
                x0 = np.clip(x0, -5, 5)

            fx0 = counted_func(x0)
            if fx0 < best_f:
                best_x, best_f = nelder_mead(x0, fx0, budget)
            else:
                # Still try to improve from this point
                x_new, f_new = nelder_mead(x0, fx0, budget)
                if f_new < best_f:
                    best_x, best_f = x_new, f_new

        # Final polish with smaller step size around best
        if calls[0] < max_calls and time.time() < deadline - 5:
            x_polish = best_x.copy()
            f_polish = best_f
            # Small local search
            for _ in range(min(50, max_calls - calls[0])):
                if time.time() > deadline:
                    break
                dx = rng.standard_normal(dim) * 0.1
                x_try = np.clip(x_polish + dx, -5, 5)
                f_try = counted_func(x_try)
                if f_try < f_polish:
                    x_polish, f_polish = x_try, f_try
            if f_polish < best_f:
                best_x, best_f = x_polish, f_polish

        return best_x, best_f
# EVOLVE-BLOCK-END
