"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Multi-start with different initializations
        # Use a portion of budget for global exploration, rest for local refinement
        n_starts = 5
        per_start_budget = budget // n_starts
        # Keep some budget for final refinement
        final_budget = 50

        for s in range(n_starts):
            if time.time() > deadline - 30:
                break

            # Different initialization strategies
            if s == 0:
                x = np.zeros(dim)
            elif s == 1:
                x = rng.uniform(-5, 5, dim)
            elif s == 2:
                x = rng.uniform(-3, 3, dim)
            elif s == 3:
                x = rng.uniform(-2, 2, dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with adaptive sigma
            sigma = 1.0
            success_count = 0
            eval_count = 0

            for _ in range(per_start_budget):
                if time.time() > deadline - 30:
                    break

                # Adaptive sigma based on success rate (1/5 rule variant)
                if success_count > 10:
                    success_rate = success_count / (success_count + 10)
                    if success_rate > 0.35:
                        sigma *= 1.3
                    elif success_rate < 0.15:
                        sigma *= 0.7
                    # Keep sigma in reasonable bounds
                    sigma = np.clip(sigma, 0.01, 10.0)

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    pass  # sigma adjustment handled above

            # Local refinement around best point found in this start
            if time.time() > deadline - 30:
                break

            # Use a smaller sigma for local refinement
            sigma_local = 0.1
            x_local = x.copy()
            fx_local = fx

            for _ in range(min(20, per_start_budget // 5)):
                if time.time() > deadline - 30:
                    break

                y = x_local + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)

                if fy < fx_local:
                    x_local, fx_local = y, fy
                    if fx_local < best_f:
                        best_f = fx_local
                        best_x = x_local.copy()
                    sigma_local *= 1.2
                else:
                    sigma_local *= 0.8

        # Final global refinement around the best point found
        if best_x is not None and time.time() < deadline - 10:
            x_final = best_x.copy()
            fx_final = best_f
            sigma_final = 0.5

            for _ in range(final_budget):
                if time.time() > deadline - 5:
                    break

                y = x_final + sigma_final * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)

                if fy < fx_final:
                    x_final, fx_final = y, fy
                    if fx_final < best_f:
                        best_f = fx_final
                        best_x = x_final.copy()
                    sigma_final *= 1.1
                else:
                    sigma_final *= 0.9

                # Adaptive adjustment
                if sigma_final > 5.0:
                    sigma_final = 5.0
                elif sigma_final < 0.005:
                    sigma_final = 0.005

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
