"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Stochastic gradient estimation with persistent direction for sharp ridge."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initial random points to find a good starting region
        best_x = None
        best_f = np.inf
        n_init = 5
        for i in range(n_init):
            x = rng.uniform(-4.5, 4.5, d)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        x = best_x
        fx = best_f
        calls_made = n_init

        # Adaptive step size
        step_size = 0.5
        min_step = 1e-6
        max_step = 2.0

        # Persistent direction for ridge tracking
        direction = np.zeros(d)

        # Phase 1: Stochastic gradient estimation with persistent direction
        # Use finite differences to estimate gradient along a persistent direction
        # and orthogonal random perturbations for cross-ridge minimization

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Estimate gradient using finite differences along current direction
            # and random orthogonal directions
            eps = max(step_size * 0.5, 1e-4)

            # Try moving along the persistent direction
            if np.linalg.norm(direction) > 1e-10:
                dir_norm = direction / np.linalg.norm(direction)
                x_plus = np.clip(x + eps * dir_norm, -5, 5)
                x_minus = np.clip(x - eps * dir_norm, -5, 5)
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls_made += 2

                grad_dir = (f_plus - f_minus) / (2 * eps)

                # Move in the negative gradient direction
                move = -grad_dir * step_size * dir_norm
                x_new = np.clip(x + move, -5, 5)
                f_new = func(x_new)
                calls_made += 1

                if f_new < fx:
                    # Update persistent direction based on what worked
                    direction = 0.8 * direction + 0.2 * dir_norm
                    x, fx = x_new, f_new
                    step_size = min(step_size * 1.2, max_step)
                else:
                    # Try orthogonal perturbation
                    ortho = rng.standard_normal(d)
                    ortho -= np.dot(ortho, dir_norm) * dir_norm
                    ortho_norm = np.linalg.norm(ortho)
                    if ortho_norm > 1e-10:
                        ortho /= ortho_norm
                        x_ortho = np.clip(x + step_size * 0.5 * ortho, -5, 5)
                        f_ortho = func(x_ortho)
                        calls_made += 1

                        if f_ortho < fx:
                            x, fx = x_ortho, f_ortho
                            step_size = min(step_size * 1.1, max_step)
                            # Blend direction towards orthogonal
                            direction = 0.7 * direction + 0.3 * ortho
                        else:
                            step_size = max(step_size * 0.8, min_step)
                    else:
                        step_size = max(step_size * 0.8, min_step)

            # If direction is near zero, initialize it
            if np.linalg.norm(direction) < 1e-10:
                direction = rng.standard_normal(d)
                direction /= np.linalg.norm(direction)

        # Phase 2: Local polishing with Nelder-Mead if we have time
        if calls_made < budget and time.time() < deadline:
            from scipy.optimize import minimize

            # Use remaining budget for local optimization
            remaining = budget - calls_made
            if remaining > 10:
                try:
                    result = minimize(
                        func,
                        x,
                        method='Nelder-Mead',
                        options={'maxiter': min(remaining, 200), 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    if result.fun < fx:
                        x, fx = result.x, result.fun
                except Exception:
                    pass

        return x, fx
# EVOLVE-BLOCK-END
