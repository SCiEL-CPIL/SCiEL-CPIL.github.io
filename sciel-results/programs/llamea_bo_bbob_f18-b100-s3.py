"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with covariance adaptation, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        best_x = None
        best_f = np.inf

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        total_budget = self.budget
        remaining = total_budget

        # Phase 1: Multi-start with (1+lambda)-ES
        # We'll do a few restarts with different initial points
        max_restarts = 5
        restarts_done = 0

        while remaining > 0 and restarts_done < max_restarts and time.time() < deadline:
            # Choose initial point
            if restarts_done == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4, 4, self.dim)

            fx = func(x)
            remaining -= 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            if remaining <= 0:
                break

            # (1+lambda)-ES with simple adaptation
            lam = 8
            sigma = 1.0
            # Simple covariance: start with identity, adapt based on successes
            C = np.eye(self.dim)
            # Track step sizes per dimension
            step_sizes = np.ones(self.dim)

            # Adaptive parameters
            chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(20*self.dim**2))
            c_sigma = (1 + 2/min(1/(lam+1), (self.dim-1)/(self.dim+3))) / (self.dim + 8)
            d_sigma = 1 + 2*max(0, sqrt(self.dim)/chi_n - 1) + c_sigma
            h_sigma = 1.4 + 2/(self.dim + 4)

            # Cumulative step
            ps = np.zeros(self.dim)
            # Path length control
            sigma_path = 0.0

            # Success history for step size adaptation
            success_window = []
            window_size = 10

            iters_in_restart = 0
            max_iters_per_restart = max(10, remaining // (max_restarts - restarts_done))
            # But we also need to respect the total remaining budget
            max_iters_per_restart = min(max_iters_per_restart, remaining)

            for _ in range(max_iters_per_restart):
                if remaining <= 0:
                    break
                if time.time() > deadline - 2.0:
                    break

                iters_in_restart += 1

                # Generate lambda offspring
                offspring = []
                offspring_f = []

                for j in range(lam):
                    # Sample from N(0, C)
                    # For simplicity, use diagonal adaptation
                    z = rng.standard_normal(self.dim)
                    # Apply step sizes
                    z = z * step_sizes
                    y = np.clip(x + sigma * z, -5, 5)
                    fy = func(y)
                    remaining -= 1
                    offspring.append(y)
                    offspring_f.append(fy)

                    if remaining <= 0:
                        break

                if len(offspring) == 0:
                    break

                # Select best offspring
                offspring_f = np.array(offspring_f)
                best_idx = np.argmin(offspring_f)
                y_best = offspring[best_idx]
                fy_best = offspring_f[best_idx]

                # Update path
                if fy_best < fx:
                    # Success
                    d = (y_best - x) / sigma
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * lam) * d
                    sigma_path = np.linalg.norm(ps)
                    # Success
                    success_window.append(1)
                    x = y_best
                    fx = fy_best
                    # Increase sigma slightly
                    sigma *= 1.1
                else:
                    ps = (1 - c_sigma) * ps
                    sigma_path = np.linalg.norm(ps)
                    success_window.append(0)
                    # Decrease sigma
                    sigma *= 0.9

                # Adapt step sizes based on success rate
                if len(success_window) >= window_size:
                    recent = success_window[-window_size:]
                    success_rate = np.mean(recent)
                    if success_rate > 0.5:
                        # Good progress, increase step sizes
                        step_sizes *= 1.05
                    elif success_rate < 0.2:
                        # Poor progress, decrease step sizes
                        step_sizes *= 0.9
                    # Clip step sizes
                    step_sizes = np.clip(step_sizes, 0.1, 10.0)
                    success_window = success_window[-window_size:]

                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Check for stagnation
                if iters_in_restart > 20 and sigma < 0.01:
                    break

                # Check if we're making progress
                if iters_in_restart % 50 == 0 and sigma < 0.05:
                    break

            restarts_done += 1

        # Phase 2: Local refinement around best point
        if remaining > 10 and time.time() < deadline - 5.0:
            # Do a fine-grained search around best_x
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(remaining, 50)):
                if remaining <= 0 or time.time() > deadline - 1.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining -= 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
                if sigma < 1e-4:
                    break

        return best_x, best_f


def sqrt(x):
    return np.sqrt(x)
# EVOLVE-BLOCK-END
