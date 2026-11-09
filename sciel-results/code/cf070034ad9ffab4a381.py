"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal step-size ES with success-rule adaptation and periodic global restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Diagonal step sizes (one per dimension)
        sigma = np.ones(dim) * 1.0

        # Success counters for 1/5 rule per dimension
        success_count = np.zeros(dim)
        trial_count = np.zeros(dim)
        success_window = 20  # evaluate adaptation every 20 trials per dim

        # Parameters for global restarts
        restart_interval = 80  # check for restart every 80 calls
        restart_threshold = 5.0  # if improvement < this over restart_interval, consider restart
        last_recheck_fx = best_fx
        last_recheck_call = calls

        while calls < self.budget and time.time() < deadline:
            # Generate a perturbation
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()

            # Track success per dimension
            for d in range(dim):
                if abs(y[d] - x[d]) > 1e-12:
                    trial_count[d] += 1
                    if fy < fx:
                        success_count[d] += 1

            if fy < fx:
                # Accept step
                x = y.copy()
                fx = fy
                # Slightly increase sigma on success
                sigma *= 1.05
            else:
                # Reject step, slightly decrease sigma
                sigma *= 0.95

            # Periodic adaptation of sigma using 1/5 rule
            for d in range(dim):
                if trial_count[d] >= success_window:
                    rate = success_count[d] / trial_count[d]
                    if rate > 0.5:
                        sigma[d] *= 1.2
                    elif rate < 0.2:
                        sigma[d] *= 0.8
                    success_count[d] = 0
                    trial_count[d] = 0

            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Periodic global restart check
            if calls - last_recheck_call >= restart_interval:
                improvement = last_recheck_fx - best_fx
                if improvement < restart_threshold:
                    # Consider a global restart with some probability
                    if rng.random() < 0.3:
                        x = rng.uniform(-5, 5, dim)
                        fx = func(x)
                        calls += 1
                        sigma = np.ones(dim) * 1.0
                        success_count = np.zeros(dim)
                        trial_count = np.zeros(dim)
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                last_recheck_fx = best_fx
                last_recheck_call = calls

            if time.time() > deadline:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
