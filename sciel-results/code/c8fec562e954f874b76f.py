# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid SGD and (1+1)-ES optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def _check_deadline():
            return time.time() < deadline

        # Phase 1: Multi-start exploration
        # Evaluate origin and a few random points to find a good basin
        n_starts = 6
        starts = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        for s in starts:
            if not _check_deadline():
                break
            if call_count >= budget:
                break
            x = np.clip(s.copy(), -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        x = best_x.copy()
        fx = best_f

        # Phase 2: Stochastic Gradient Descent (SGD) with finite differences
        # Weierstrass is smooth, so gradient info is valuable.
        h = 1e-5  # Step size for finite differences
        alpha = 0.1  # Initial learning rate
        max_sgd_steps = 30  # Limit steps to save budget for polish
        sgd_calls_per_step = 2 * dim  # Central differences

        for step in range(max_sgd_steps):
            if not _check_deadline():
                break
            if call_count + sgd_calls_per_step >= budget:
                break

            # Estimate gradient using central differences
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                call_count += 2
                
                grad[i] = (f_plus - f_minus) / (2 * h)

            # Update x
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break  # Gradient is zero, likely at a minimum

            # Normalize gradient to control step size relative to magnitude
            # Or just use alpha directly. Let's use alpha directly but adapt it.
            x_new = x - alpha * grad
            x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            call_count += 1

            if f_new < fx:
                x = x_new
                fx = f_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                # Increase learning rate slightly if we are making progress
                alpha *= 1.1
            else:
                # Decrease learning rate if we overshoot or go uphill
                alpha *= 0.5

            # Clamp alpha
            alpha = max(1e-6, min(alpha, 1.0))

        # Phase 3: Fine-tuning with (1+1)-ES
        # Use remaining budget for local search with small steps
        if not _check_deadline() and call_count < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05  # Small initial sigma for polish
            remaining = budget - call_count

            for _ in range(remaining):
                if not _check_deadline():
                    break
                if call_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.05, 1.0)
                else:
                    sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
