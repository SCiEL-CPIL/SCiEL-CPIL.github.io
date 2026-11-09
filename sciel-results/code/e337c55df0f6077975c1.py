"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """
    Discrete lattice search + local Nelder-Mead refinement with deadline guard.
    Strategy:
      1. Evaluate a 3^5 = 243 point lattice over [-5,5]^5 to find the best basin.
      2. Use remaining budget for local Nelder-Mead refinement from the best lattice point.
      3. Deadline guard: stop if wall-clock exceeds 280s (20s safety margin).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        # Phase 1: 3^5 lattice search (243 evaluations)
        # Lattice points: -5, 0, 5 in each dimension
        lattice_vals = np.array([-5.0, 0.0, 5.0])
        lattice_points = np.array(np.meshgrid(*[lattice_vals] * dim)).T.reshape(-1, dim)
        n_lattice = len(lattice_points)  # 243

        if n_lattice > budget:
            # Fallback: random sampling if budget is too small (shouldn't happen with budget=500)
            n_lattice = budget - 1
            lattice_points = np.random.default_rng(0).uniform(-5, 5, (n_lattice, dim))

        best_x = None
        best_f = np.inf

        # Evaluate lattice points
        for i in range(n_lattice):
            if time.time() > deadline:
                break
            if budget - (i + 1) < 10:  # reserve some budget for refinement
                break

            x = lattice_points[i]
            fx = func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        calls_made = i + 1 if 'i' in dir() else 0
        remaining = budget - calls_made

        if best_x is None:
            # No evaluations made, do a single random eval
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            calls_made += 1
            remaining = budget - calls_made

        # Phase 2: Local refinement using Nelder-Mead from best lattice point
        # Use remaining budget, but Nelder-Mead doesn't directly support budget limits.
        # We'll wrap the function to count calls and stop when budget is exhausted.
        # Alternative: use a simple (1+1)-ES for refinement with budget control.

        # Let's use (1+1)-ES for refinement with adaptive step size
        if remaining > 0:
            rng = np.random.default_rng(1)
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with smaller step for refinement

            for _ in range(remaining):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85

                # Early stop if we're very close to boundary and not improving
                if sigma < 1e-6:
                    break

            best_x, best_f = x, fx

        return best_x, best_f
# EVOLVE-BLOCK-END
