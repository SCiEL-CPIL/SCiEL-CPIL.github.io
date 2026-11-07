"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES restarts with simple adaptive step size.

    Strategy:
      1. Multiple (1+1)-ES restarts with diverse random initializations.
      2. Simple, robust step-size adaptation (1.5x on success, 0.87x on failure).
      3. Equal budget allocation for restarts to ensure diversity.
      4. Final local refinement phase using the best point found.
      5. Explicit wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Reserve calls for final refinement
        reserve = min(50, budget // 5)
        
        # Phase 1: Portfolio of (1+1)-ES restarts
        n_restarts = 4
        alloc_per_restart = (budget - reserve) // n_restarts
        
        # Diverse initializations
        init_points = [
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
        ]
        
        # Diverse initial sigmas
        initial_sigmas = [2.0, 1.5, 3.0, 1.0]

        for r in range(n_restarts):
            if call_count[0] >= budget - reserve:
                break
            if time.time() > deadline - 20:
                break

            x = init_points[r]
            fx = make_call(x)
            if fx is None:
                break
            update_best(x, fx)

            sigma = initial_sigmas[r]
            
            # Simple (1+1)-ES loop
            for _ in range(alloc_per_restart):
                if call_count[0] >= budget - reserve:
                    break
                if time.time() > deadline - 20:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.5
                else:
                    sigma *= 0.87

                # Keep sigma within bounds
                sigma = float(np.clip(sigma, 0.01, 10.0))

        # Phase 2: Final local refinement around the best point
        if best_x is not None and call_count[0] < budget - 5 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining_calls = budget - call_count[0] - 5
            
            for _ in range(remaining_calls):
                if call_count[0] >= budget - 5:
                    break
                if time.time() > deadline - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.2
                else:
                    sigma *= 0.8

                sigma = float(np.clip(sigma, 0.001, 1.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
