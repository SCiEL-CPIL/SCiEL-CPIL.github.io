"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and time guard for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s

        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if len(x) != dim:
                x = x[:dim]
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Halton/QMC global exploration (40% of budget)
        n_global = int(budget * 0.4)
        if n_global < 20:
            n_global = 20

        # Generate Halton sequence points
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_point(i, dim, bases):
            return np.array([halton(i + 1, b) for b in bases[:dim]])

        # Use prime bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        # Interleave with random for diversity
        rng = np.random.default_rng(42)
        for i in range(n_global):
            if time.time() > deadline:
                break
            if i % 3 == 0:
                # Halton point mapped to [-5, 5]
                p = halton_point(i, dim, primes)
                x = -5.0 + 10.0 * p
            else:
                # Random point
                x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is None:
                break

        # Phase 2: Local refinement with CMA-ES-like strategy on best regions
        # Use the best few points as seeds for local search
        if best_x is not None and calls_made < budget:
            # Try multiple local searches from top candidates
            # We'll do a simple adaptive step-size search from the global best
            x_cur = best_x.copy()
            f_cur = best_f
            sigma = 1.5

            # Adaptive local search with momentum
            for step in range(budget - calls_made):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Generate candidate
                direction = rng.standard_normal(dim)
                y = x_cur + sigma * direction
                fy = safe_call(y)
                if fy is None:
                    break

                if fy < f_cur:
                    # Accept and increase step slightly
                    x_cur = y.copy()
                    f_cur = fy
                    sigma = min(sigma * 1.2, 4.0)
                else:
                    # Reject and decrease step
                    sigma *= 0.85

                    # Periodically try a larger exploratory jump
                    if step % 50 == 49 and sigma < 0.3:
                        jump = rng.standard_normal(dim) * 2.0
                        yj = x_cur + jump
                        fyj = safe_call(yj)
                        if fyj is not None and fyj < f_cur:
                            x_cur = yj.copy()
                            f_cur = fyj
                            sigma = 1.0

        # Phase 3: Coordinate-wise fine-tuning if budget remains
        if best_x is not None and calls_made < budget:
            x_cur = best_x.copy()
            f_cur = best_f
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

            for ss in step_sizes:
                if calls_made >= budget or time.time() > deadline:
                    break
                improved = True
                while improved:
                    improved = False
                    if calls_made >= budget or time.time() > deadline:
                        break
                    for d in range(dim):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        for sign in [1, -1]:
                            if calls_made >= budget:
                                break
                            y = x_cur.copy()
                            y[d] += sign * ss
                            fy = safe_call(y)
                            if fy is None:
                                break
                            if fy < f_cur:
                                x_cur = y.copy()
                                f_cur = fy
                                improved = True
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
