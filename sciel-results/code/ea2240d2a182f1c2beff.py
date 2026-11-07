"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid scout + Powell optimizer with ES refinement for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 300s limit

        # Track best solution and function value
        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            if call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Scout with several starting points to find a good basin
        # Rosenbrock has a valley; the minimum is near (1,...,1) in standard form,
        # but hidden shift/rotation moves it. Try multiple starting points.
        rng = np.random.default_rng(42)

        # Candidate starting points: origin, nominal min, and random points
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim),
            np.array([1.0] * self.dim),
        ]
        # Add some random starts spread across the domain
        for _ in range(6):
            starts.append(rng.uniform(-4.5, 4.5, self.dim))

        # Evaluate all starting points
        for s in starts:
            if call_count >= self.budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            f_val = wrapped(s_clipped)

        # Phase 2: Run Powell's method from the best scout point
        if best_x is not None:
            # Powell's method with tight tolerances
            result = minimize(
                wrapped,
                best_x,
                method='Powell',
                options={
                    'maxiter': 150,
                    'maxfev': max(1, self.budget - call_count - 80),  # reserve budget for later phases
                    'xtol': 1e-10,
                    'ftol': 1e-12,
                }
            )
            # Update best
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5.0, 5.0)

            # Phase 2.5: Valley direction line search - exploit Rosenbrock structure
            # In Rosenbrock, the valley direction is approximately along the diagonal
            # For shifted/rotated case, try several candidate directions
            if call_count < self.budget - 50 and time.time() < deadline:
                x_valley = best_x.copy()
                fx_valley = best_f
                # Try several candidate valley directions
                candidate_dirs = [
                    np.ones(self.dim) / np.sqrt(self.dim),
                    np.array([1.0, 1.0, 0.5, 0.25, 0.125]) / np.linalg.norm([1.0, 1.0, 0.5, 0.25, 0.125]),
                    np.array([0.5, 1.0, 1.0, 0.5, 0.25]) / np.linalg.norm([0.5, 1.0, 1.0, 0.5, 0.25]),
                    rng.standard_normal(self.dim) / np.sqrt(self.dim),
                ]
                for d in candidate_dirs:
                    if call_count >= self.budget - 30 or time.time() > deadline:
                        break
                    # Line search along direction d from current best
                    for alpha in [0.1, 0.05, 0.02, 0.01, -0.01, -0.02, -0.05, -0.1]:
                        if call_count >= self.budget - 10 or time.time() > deadline:
                            break
                        x_try = np.clip(x_valley + alpha * d, -5.0, 5.0)
                        f_try = wrapped(x_try)
                        if f_try < fx_valley:
                            x_valley = x_try.copy()
                            fx_valley = f_try
                            if fx_valley < best_f:
                                best_f = fx_valley
                                best_x = x_valley.copy()

            # Phase 3: (1+1)-ES with adaptive sigma for refinement
            if call_count < self.budget and time.time() < deadline:
                x_es = best_x.copy()
                fx_es = best_f
                sigma = 0.15  # Start with moderate step
                es_calls = 0
                max_es_calls = min(80, self.budget - call_count - 40)
                
                for _ in range(max_es_calls):
                    if call_count >= self.budget - 20 or time.time() > deadline:
                        break
                    y = np.clip(x_es + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = wrapped(y)
                    es_calls += 1
                    if fy < fx_es:
                        x_es, fx_es = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
                    
                    if fx_es < best_f:
                        best_f = fx_es
                        best_x = x_es.copy()

            # Phase 4: Final high-precision polish with Nelder-Mead
            if call_count < self.budget and time.time() < deadline:
                result2 = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 80,
                        'maxfev': max(1, self.budget - call_count - 20),
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                    }
                )
                if result2.fun < best_f:
                    best_f = result2.fun
                    best_x = np.clip(result2.x, -5.0, 5.0)

            # Phase 5: Final micro-ES polish with very small steps
            if call_count < self.budget and time.time() < deadline:
                x_es = best_x.copy()
                fx_es = best_f
                sigma = 0.01  # Very small step for final polish
                max_es_calls = min(30, self.budget - call_count)
                
                for _ in range(max_es_calls):
                    if call_count >= self.budget or time.time() > deadline:
                        break
                    y = np.clip(x_es + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = wrapped(y)
                    if fy < fx_es:
                        x_es, fx_es = y, fy
                        sigma *= 1.1
                    else:
                        sigma *= 0.95
                    
                    if fx_es < best_f:
                        best_f = fx_es
                        best_x = x_es.copy()

        # Ensure we have a valid return
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
