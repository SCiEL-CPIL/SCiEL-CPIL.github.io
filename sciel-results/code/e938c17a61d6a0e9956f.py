"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # We'll do multiple restarts with decreasing exploration
        # Phase 1: broad exploration with larger sigma
        # Phase 2: refinement

        best_x = None
        best_f = np.inf
        calls = 0

        # Number of restarts: use budget to determine
        # Each restart uses a fraction of the budget
        # Let's do 3 phases:
        # Phase 1: 40% budget, sigma=3.0, 5 restarts
        # Phase 2: 35% budget, sigma=1.5, 5 restarts  
        # Phase 3: 25% budget, sigma=0.5, 5 restarts

        phases = [
            (0.40, 3.0, 5),
            (0.35, 1.5, 5),
            (0.25, 0.5, 5),
        ]

        for frac, base_sigma, n_restarts in phases:
            phase_budget = max(10, int(budget * frac))
            per_restart = max(2, phase_budget // n_restarts)

            for r in range(n_restarts):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Initialize near best if we have one, else random
                if best_x is not None and r > 0:
                    x = best_x + base_sigma * 0.3 * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()

                sigma = base_sigma
                # 1/5 success rule parameters
                success_count = 0
                window = 10

                for i in range(per_restart - 1):
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break

                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1

                    if fy < fx:
                        x, fx = y, fy
                        success_count += 1
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                    else:
                        pass

                    # Adapt sigma based on success rate over window
                    if (i + 1) % window == 0:
                        rate = success_count / window
                        if rate > 0.5:
                            sigma *= 1.3
                        elif rate < 0.3:
                            sigma *= 0.7
                        success_count = 0

                    # Keep sigma in reasonable range
                    sigma = max(0.01, min(sigma, 10.0))

                if time.time() > deadline:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
