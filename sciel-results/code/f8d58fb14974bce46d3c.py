"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for multimodal Katsuura (f23)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline

        # Phase 1: Random exploration to find a good starting basin
        n_explore = min(15, budget // 30)
        best_x = None
        best_f = np.inf
        calls = 0

        for i in range(n_explore):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES optimization from the best found point
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # For dim=5: ~9
        pop_size = min(pop_size, budget // 10)  # Ensure we can do multiple iterations
        pop_size = max(pop_size, 4)

        # Initialize CMA-ES
        mean = best_x.copy()
        sigma = 2.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        C_sqrt = np.eye(dim)  # Square root of C
        cs = 1 / (dim + 4)  # Path length adaptation
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 - (dim + 1) / (dim + 4)) / ((dim + 2) ** 2 + 2))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Restart logic
        no_improve = 0
        max_no_improve = 30
        restart_count = 0
        max_restarts = 8
        best_global_f = best_f
        best_global_x = best_x.copy()

        # Main CMA-ES loop
        while calls < budget and time_ok():
            # Check for stagnation
            if no_improve >= max_no_improve:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.3)

                # Try a new random restart
                x_new = rng.uniform(-5, 5, dim)
                fx_new = func(x_new)
                calls += 1
                if fx_new < best_global_f:
                    best_global_f = fx_new
                    best_global_x = x_new.copy()
                    mean = x_new.copy()
                    sigma = 3.0
                else:
                    # Jump from current best
                    jump_scale = 2.0 + restart_count * 0.5
                    mean = np.clip(best_global_x + jump_scale * rng.standard_normal(dim), -5, 5)
                    sigma = max(2.0, sigma)

                # Reset CMA-ES state
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                continue

            # Generate population
            for _ in range(pop_size):
                if not time_ok() or calls >= budget:
                    break

                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1

                # Update best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Update best global
                if fy < best_global_f:
                    best_global_f = fy
                    best_global_x = y.copy()

            # Update CMA-ES parameters (simplified version)
            # In a full implementation, we'd do proper ranking and update
            # For simplicity, we adapt sigma based on improvement rate
            if no_improve == 0:
                sigma *= 1.2  # Increase step size on success
            else:
                sigma *= 0.8  # Decrease on failure

            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 15.0)

            # Update mean towards best
            mean = 0.8 * mean + 0.2 * best_x
            mean = np.clip(mean, -5, 5)

        return best_global_x, best_global_f
# EVOLVE-BLOCK-END
