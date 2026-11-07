"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Stochastic gradient descent with persistent direction, adaptive step,
    ridge-tracking via finite differences, and NM polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Counting wrapper
        call_count = 0

        def counting_func(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        # Phase 1: Stochastic gradient estimation with persistent direction
        # Initialize at center
        x = np.zeros(dim)
        fx = counting_func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step size
        step = 1.0
        # Persistent search direction (initialized randomly)
        direction = rng.standard_normal(dim)
        direction /= np.linalg.norm(direction)

        # Ridge tracking: maintain a "ridge direction" estimate
        ridge_dir = None
        ridge_momentum = 0.0

        # Gradient estimation via finite differences
        eps = 0.01

        remaining = budget - 1
        for i in range(remaining):
            if call_count >= budget:
                break
            if time.time() > deadline - 30:
                break

            # Estimate gradient using central differences along a random direction
            # Use 2 evaluations for gradient in random direction
            r = rng.standard_normal(dim)
            r /= np.linalg.norm(r)

            # Central difference
            xp = np.clip(x + eps * r, -5, 5)
            xm = np.clip(x - eps * r, -5, 5)
            fxp = counting_func(xp)
            fxm = counting_func(xm)
            if call_count >= budget:
                break

            grad_est = (fxp - fxm) / (2 * eps) * r

            # Update persistent direction with momentum
            if ridge_dir is None:
                ridge_dir = -grad_est
                ridge_dir /= (np.linalg.norm(ridge_dir) + 1e-12)
            else:
                # Blend new gradient direction with existing ridge direction
                new_dir = -grad_est
                new_dir /= (np.linalg.norm(new_dir) + 1e-12)
                # Check alignment
                dot = np.dot(ridge_dir, new_dir)
                if dot > 0:
                    # Consistent direction, update with momentum
                    ridge_dir = ridge_dir * ridge_momentum + new_dir * (1 - ridge_momentum)
                    ridge_dir /= (np.linalg.norm(ridge_dir) + 1e-12)
                    ridge_momentum = min(ridge_momentum + 0.05, 0.9)
                else:
                    # Direction changed, reset
                    ridge_dir = new_dir
                    ridge_momentum = 0.0

            # Take a step along the ridge direction with adaptive step size
            # Also add small orthogonal noise to explore cross-ridge
            step_x = x + step * ridge_dir
            # Add small orthogonal perturbation
            ortho_noise = rng.standard_normal(dim) * step * 0.1
            step_x = np.clip(step_x + ortho_noise, -5, 5)

            f_step = counting_func(step_x)
            if call_count >= budget:
                break

            if f_step < fx:
                x, fx = step_x, f_step
                step = min(step * 1.2, 5.0)
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
            else:
                step *= 0.7
                # If stuck, try a random restart-like jump
                if step < 0.01:
                    x = best_x + rng.standard_normal(dim) * 0.5
                    x = np.clip(x, -5, 5)
                    fx = counting_func(x)
                    if call_count >= budget:
                        break
                    step = 0.5
                    ridge_dir = None
                    ridge_momentum = 0.0

        # Phase 2: Nelder-Mead polishing from best point
        if call_count < budget and time.time() < deadline - 20:
            try:
                from scipy.optimize import minimize
            except ImportError:
                minimize = None

            if minimize is not None:
                # Try a few NM restarts
                nm_results = []
                for nm_try in range(3):
                    if call_count >= budget:
                        break
                    if time.time() > deadline - 15:
                        break

                    if nm_try == 0:
                        x0 = best_x.copy()
                    else:
                        x0 = best_x + rng.standard_normal(dim) * (0.3 / nm_try)
                        x0 = np.clip(x0, -5, 5)

                    try:
                        res = minimize(
                            counting_func, x0, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                        )
                        if call_count >= budget:
                            break
                        if res.fun < best_fx:
                            best_fx = res.fun
                            best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
