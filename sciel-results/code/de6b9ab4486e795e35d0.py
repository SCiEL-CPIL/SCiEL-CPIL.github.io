"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and a final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def safe_call(x):
            nonlocal calls_made
            try:
                fx = func(x)
                calls_made += 1
                if not np.isfinite(fx):
                    return np.inf
                return float(fx)
            except Exception:
                calls_made += 1
                return np.inf

        # Phase 1: (1+1)-ES with restarts
        # We split budget into chunks; if no improvement for a while, restart from best
        # Use a success-based sigma adaptation

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = safe_call(x)
        if fx < best_f:
            best_f, best_x = fx, x.copy()

        sigma = 2.0
        no_improve = 0
        max_no_improve = 50
        chunk_calls = 0
        max_chunk = 80

        while calls_made < budget and time.time() < deadline:
            # Check time
            if time.time() > deadline - 5.0:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = safe_call(y)

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                sigma *= 1.3
                chunk_calls += 1
            else:
                no_improve += 1
                sigma *= 0.85
                chunk_calls += 1

            # Restart if stuck
            if no_improve >= max_no_improve or chunk_calls >= max_chunk:
                # Restart from best
                x = best_x.copy()
                fx = best_f
                sigma = max(0.1, sigma * 0.5)
                no_improve = 0
                chunk_calls = 0

        # Phase 2: Polish - use Nelder-Mead or a fine local search
        # Use scipy if available for a quick polish
        try:
            from scipy.optimize import minimize

            if best_x is not None and calls_made < budget and time.time() < deadline - 2.0:
                # Use Nelder-Mead with tight settings
                remaining = budget - calls_made
                if remaining > 5:
                    def wrapper(z):
                        nonlocal calls_made
                        if calls_made >= budget:
                            return best_f
                        if time.time() > deadline - 1.0:
                            return best_f
                        zc = np.clip(z, -5, 5)
                        return safe_call(zc)

                    # Nelder-Mead
                    res = minimize(wrapper, best_x, method='Nelder-Mead',
                                   options={'maxiter': min(remaining, 100),
                                            'xatol': 1e-8, 'fatol': 1e-10})
                    if res.success and res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
        except Exception:
            pass

        # Phase 3: Final fine-grained local search if budget remains
        if best_x is not None and calls_made < budget and time.time() < deadline - 1.0:
            # Adaptive local search with decreasing step
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            for _ in range(min(50, budget - calls_made)):
                if time.time() > deadline - 0.5:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
