"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid multi-start scout + Powell/Nelder-Mead polish + (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Wrap func to count calls and track best
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def wrapped(x):
            call_count[0] += 1
            fx = func(x)
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start Powell with diverse initial points
        # Use a mix of random and structured starting points
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Corners of [-4, 4]^dim (since minimizer is in [-4,4])
        for i in range(8):
            pt = rng.uniform(-4, 4, dim)
            start_points.append(pt)
        # Some structured points
        for i in range(4):
            pt = rng.uniform(-5, 5, dim)
            start_points.append(pt)

        # Evaluate all starting points
        for sp in start_points:
            sp = np.clip(sp, -5, 5)
            wrapped(sp)

        # Phase 2: Powell from the best starting points
        # Sort by function value and run Powell on top candidates
        candidates = []
        for sp in start_points:
            sp = np.clip(sp, -5, 5)
            if len(candidates) == 0 or wrapped(sp) < wrapped(candidates[-1]):
                candidates.append(sp)
        candidates = sorted(candidates, key=lambda x: wrapped(x))

        # Run Powell on top 3 candidates
        for i, x0 in enumerate(candidates[:3]):
            if time.time() > deadline - 30:
                break
            if call_count[0] >= budget - 50:
                break
            try:
                res = minimize(
                    wrapped,
                    x0,
                    method='Powell',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-10}
                )
            except Exception:
                continue

        # Phase 3: Nelder-Mead from the current best
        if time.time() < deadline - 20 and call_count[0] < budget - 30:
            if best_x[0] is not None:
                try:
                    res = minimize(
                        wrapped,
                        best_x[0],
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                except Exception:
                    pass

        # Phase 4: (1+1)-ES refinement with adaptive step size
        if time.time() < deadline - 10 and call_count[0] < budget - 10:
            if best_x[0] is not None:
                x = best_x[0].copy()
                fx = best_f[0]
                sigma = 0.5
                remaining = budget - call_count[0]
                for _ in range(remaining):
                    if time.time() > deadline - 5:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = wrapped(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.3
                    else:
                        sigma *= 0.85
                    if sigma < 1e-10:
                        break

        # Final check: ensure we haven't exceeded budget
        # Return best found
        if best_x[0] is not None:
            return best_x[0], best_f[0]
        else:
            return np.zeros(dim), 0.0
# EVOLVE-BLOCK-END
