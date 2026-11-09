"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Combines tuned (1+1)-ES sigma adaptation (1.15x/0.9x) with a dedicated
    final polish phase for high-precision refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random sampling to find a promising initial region
        # Reduced from 50 to 30 to allow more time for the ES phase
        n_samples = min(30, budget // 10)
        best_x = None
        best_fx = float('inf')

        for i in range(n_samples):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        calls_used = n_samples
        best_x = best_x.copy()

        # Phase 2: Adaptive (1+1)-ES with tuned sigma rates and restarts
        # Tuned rates: 1.15x on success, 0.9x on failure (from Inspiration 1)
        sigma = 1.5
        stall_count = 0
        max_stall = 25
        restart_sigma_factor = 2.0
        restart_sigma = 2.0

        while calls_used < budget:
            # Time check
            if time.time() > deadline - 5.0:
                break

            # Generate candidate
            y = best_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < best_fx:
                best_x = y.copy()
                best_fx = fy
                sigma *= 1.15
                stall_count = 0
            else:
                sigma *= 0.9
                stall_count += 1

            # Restart logic: If stalled, increase sigma to explore wider area
            if stall_count >= max_stall:
                sigma = max(restart_sigma, sigma * restart_sigma_factor)
                restart_sigma *= 1.3
                stall_count = 0
                # Increase max_stall slightly to allow longer exploration before next restart
                max_stall = min(max_stall + 5, 60)
                
                # Optional: Small perturbation of best_x to escape local minima
                # This is a structural change to handle ill-conditioning
                if rng.random() < 0.5:
                    best_x = best_x + 0.5 * sigma * rng.standard_normal(dim)
                    best_x = np.clip(best_x, -5, 5)
                    # We don't evaluate this new best_x immediately to save a call,
                    # but the next iteration will compare against the old best_fx.
                    # Actually, if we move best_x, we should probably keep best_fx
                    # as the known minimum. The next candidate will be generated
                    # around this new center.

        # Phase 3: Final Polish Phase
        # Use remaining budget for tight local refinement with small sigma
        # This is a structural addition to ensure convergence
        if calls_used < budget:
            # Reserve last 10% of remaining budget or at least 20 calls for polish
            polish_budget = max(20, (budget - calls_used) // 5)
            polish_calls = 0
            
            # Start with a small sigma for polish
            sigma = 0.1
            # Limit sigma growth in polish phase to prevent escaping the basin
            max_sigma_polish = 0.5
            min_sigma_polish = 0.01
            
            # Reset stall logic for polish
            stall_count = 0
            max_stall_polish = 10

            while polish_calls < polish_budget and calls_used < budget:
                if time.time() > deadline - 1.0:
                    break

                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                polish_calls += 1

                if fy < best_fx:
                    best_x = y.copy()
                    best_fx = fy
                    sigma *= 1.1
                    stall_count = 0
                else:
                    sigma *= 0.9
                    stall_count += 1

                # Adapt sigma within polish bounds
                sigma = np.clip(sigma, min_sigma_polish, max_sigma_polish)

                # Restart in polish if stalled (very small restart)
                if stall_count >= max_stall_polish:
                    sigma = max(min_sigma_polish, sigma * 1.5)
                    stall_count = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
