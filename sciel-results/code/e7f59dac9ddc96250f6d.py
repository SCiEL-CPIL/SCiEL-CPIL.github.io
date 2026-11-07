"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with per-dimension adaptive step sizes and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        def check_time():
            return time.time() > deadline

        # Phase 1: Brief coordinate descent to find a reasonable starting point
        # Use ~10% of budget for this
        init_budget = max(10, budget // 10)
        x = np.zeros(dim)
        fx = func(x)
        num_calls = 1
        best_x = x.copy()
        best_f = fx

        # Coordinate descent: try moving along each axis
        step = 1.0
        for _ in range(init_budget - 1):
            if num_calls >= budget or check_time():
                break
            for d in range(dim):
                if num_calls >= budget or check_time():
                    break
                # Try positive and negative directions
                for direction in [1, -1]:
                    y = x.copy()
                    y[d] = np.clip(y[d] + direction * step, -5, 5)
                    fy = func(y)
                    num_calls += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                        break
                # Shrink step if no improvement found
                else:
                    step *= 0.7
                    if step < 1e-4:
                        step = 0.5  # reset to explore again

        # Phase 2: Main (1+1)-ES with improved adaptation
        x = best_x.copy()
        fx = best_f
        sigmas = np.full(dim, 0.5)  # Start smaller since we have a good start
        success_ema = np.zeros(dim)  # Exponential moving average of success per dim
        ema_alpha = 0.1  # Smoothing factor

        # Track overall success rate for global adaptation
        overall_success_ema = 0.5  # Start with 50% assumption

        for _ in range(budget - num_calls):
            if num_calls >= budget or check_time():
                break

            # Generate perturbation
            perturbation = sigmas * rng.standard_normal(dim)
            y = np.clip(x + perturbation, -5, 5)
            fy = func(y)
            num_calls += 1

            if fy < fx:
                x = y
                fx = fy
                # Update per-dimension success EMA
                for d in range(dim):
                    success_ema[d] = ema_alpha * 1.0 + (1 - ema_alpha) * success_ema[d]
                overall_success_ema = ema_alpha * 1.0 + (1 - ema_alpha) * overall_success_ema
            else:
                for d in range(dim):
                    success_ema[d] = ema_alpha * 0.0 + (1 - ema_alpha) * success_ema[d]
                overall_success_ema = ema_alpha * 0.0 + (1 - ema_alpha) * overall_success_ema

            # Adapt step sizes based on success rate
            # Target success rate ~0.33 (theoretical optimum for (1+1)-ES)
            target_success = 0.33
            for d in range(dim):
                if success_ema[d] > target_success:
                    sigmas[d] *= 1.05  # Too successful, increase step
                elif success_ema[d] < target_success * 0.5:
                    sigmas[d] *= 0.95  # Too many failures, decrease step

            # Global adaptation based on overall success
            if overall_success_ema > 0.5:
                sigmas *= 1.02
            elif overall_success_ema < 0.15:
                sigmas *= 0.98

            # Clamp step sizes
            sigmas = np.clip(sigmas, 1e-8, 5.0)

            if fy < best_f:
                best_f = fy
                best_x = y.copy()

        # Phase 3: Final high-precision polish
        remaining = budget - num_calls
        if remaining > 0 and not check_time():
            # Use very small steps for fine-tuning
            polish_sigmas = np.full(dim, 0.01)
            x_polish = best_x.copy()
            fx_polish = best_f

            for _ in range(remaining):
                if check_time():
                    break
                perturbation = polish_sigmas * rng.standard_normal(dim)
                y = np.clip(x_polish + perturbation, -5, 5)
                fy = func(y)
                num_calls += 1
                if fy < fx_polish:
                    x_polish = y
                    fx_polish = fy
                    # Gradually increase step size as we converge
                    polish_sigmas *= 1.02
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                else:
                    polish_sigmas *= 0.95

                polish_sigmas = np.clip(polish_sigmas, 1e-10, 0.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
