"""Rewritten optimizer for BBOB f7 (Step ellipsoidal) with hybrid CMA-ES + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """Hybrid optimizer: CMA-ES for global search, Nelder-Mead for local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Track calls and best
        calls = [0]
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            calls[0] += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: CMA-ES with adaptive sigma
        rng = np.random.default_rng(12345)
        x = rng.uniform(-4.0, 4.0, dim)
        fx = wrapped_func(x)
        sigma = 1.5
        sigma_min = 1e-10
        sigma_max = 5.0

        # CMA-ES parameters
        pop_size = max(4, dim + 2)  # lambda
        success_count = 0
        window = 10
        target_rate = 0.2  # 1/5 rule

        # Track history for adaptation
        sigma_history = []

        # Simple CMA-ES implementation
        # Use diagonal covariance (axis-aligned) for simplicity and speed
        # with adaptive step sizes per dimension
        sigma_vec = np.ones(dim) * sigma

        # Initial population
        for phase_iter in range(200):  # max iterations for CMA-ES phase
            if calls[0] >= budget - 50:  # reserve budget for Nelder-Mead
                break
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                if calls[0] >= budget - 50:
                    break
                if time.time() > deadline:
                    break
                # Sample from multivariate normal with diagonal covariance
                z = rng.standard_normal(dim)
                y = x + sigma_vec * z
                y = np.clip(y, -5.0, 5.0)
                fy = wrapped_func(y)
                offspring.append((y, fy))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])

            # Update mean (weighted average of top half)
            mu_eff = len(offspring) // 2
            weights = np.ones(mu_eff) / mu_eff
            new_x = np.sum(np.array([offspring[i][0] for i in range(mu_eff)]) * weights[:, None], axis=0)
            new_x = np.clip(new_x, -5.0, 5.0)

            # Adapt sigma
            best_fy = offspring[0][1]
            if best_fy < fx:
                success_count += 1
            else:
                success_count = 0

            # 1/5 success rule
            if success_count > 0 and success_count % window == 0:
                sigma_vec *= 1.2
                sigma_vec = np.clip(sigma_vec, sigma_min, sigma_max)
                success_count = 0
            elif success_count == 0 and len(sigma_history) >= window:
                sigma_vec *= 0.8
                sigma_vec = np.clip(sigma_vec, sigma_min, sigma_max)

            sigma_history.append(best_fy)
            if len(sigma_history) > window:
                sigma_history.pop(0)

            x = new_x
            fx = best_fy

            # Check convergence
            if np.all(sigma_vec < 1e-8):
                break

        # Phase 2: Nelder-Mead local polish
        if calls[0] < budget and best_x is not None and time.time() < deadline:
            # Use remaining budget for Nelder-Mead
            remaining = budget - calls[0]
            # Nelder-Mead with limited iterations
            nm_result = scipy_minimize(
                wrapped_func,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': min(remaining // 2, 500),
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if nm_result.success and nm_result.fun < best_f:
                best_f = nm_result.fun
                best_x = np.clip(nm_result.x, -5.0, 5.0)

        # If we still have budget, do a final round of random restarts + polish
        if calls[0] < budget - 10 and best_x is not None and time.time() < deadline:
            # Try a few more local refinements from different starting points
            for restart in range(3):
                if calls[0] >= budget or time.time() > deadline:
                    break
                # Start from current best with small perturbation
                x_start = best_x + 0.1 * rng.standard_normal(dim)
                x_start = np.clip(x_start, -5.0, 5.0)
                nm_result = scipy_minimize(
                    wrapped_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min((budget - calls[0]) // 4, 200),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if nm_result.success and nm_result.fun < best_f:
                    best_f = nm_result.fun
                    best_x = np.clip(nm_result.x, -5.0, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
