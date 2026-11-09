"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid global-to-local optimizer for Rosenbrock with hidden transformations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None

    def __call__(self, func):
        self._start_time = time.time()
        self._call_count = 0
        dim = self.dim
        budget = self.budget

        # Wrap func to track calls and enforce deadline
        def wrapped(x):
            if self._call_count >= budget:
                return float('inf')
            self._call_count += 1
            if time.time() - self._start_time > 320:
                return float('inf')
            return func(np.asarray(x, dtype=float))

        # Generate diverse starting points
        rng = np.random.default_rng(42)

        # Priority-ordered starts: center around (1,1,1,1,1) which is standard Rosenbrock min
        # But since there's a hidden shift/rotation, we need diverse coverage
        # Start with points near the likely transformed minimum region

        starts = []

        # 1. Points near (1,...,1) with small perturbations (high priority)
        for _ in range(8):
            p = np.ones(dim) + rng.normal(0, 0.3, dim)
            p = np.clip(p, -5, 5)
            starts.append(p)

        # 2. Points in a grid around the center of the box
        for _ in range(8):
            p = rng.uniform(-2, 2, dim)
            starts.append(p)

        # 3. Broader exploration
        for _ in range(8):
            p = rng.uniform(-5, 5, dim)
            starts.append(p)

        # 4. Points near origin
        for _ in range(4):
            p = rng.normal(0, 1, dim)
            p = np.clip(p, -5, 5)
            starts.append(p)

        # 5. Additional diverse points
        for _ in range(4):
            p = rng.uniform(-4, 4, dim)
            starts.append(p)

        # Remove duplicates approximately
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.linalg.norm(s - u) < 0.5:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        starts = unique_starts

        best_x = None
        best_f = float('inf')

        # Evaluate all starting points
        for s in starts:
            if self._call_count >= budget or time.time() - self._start_time > 320:
                break
            f = wrapped(s)
            if f < best_f:
                best_f = f
                best_x = s.copy()

        if best_x is None:
            # Fallback: use first start
            best_x = starts[0].copy()
            best_f = wrapped(best_x)

        # Now do local refinement from the best point using L-BFGS-B
        remaining_budget = budget - self._call_count
        if remaining_budget > 10 and time.time() - self._start_time < 315:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max(1, remaining_budget - 5),
                             'ftol': 1e-12,
                             'gtol': 1e-8,
                             'maxls': 20}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        # If we still have budget, do a few more local refinements from other good starts
        if self._call_count < budget - 5 and time.time() - self._start_time < 310:
            # Sort starts by their function values (we only evaluated them once)
            # Do a quick Nelder-Mead from the second-best start if available
            if len(starts) > 1:
                # Try a second local optimization from a different point
                second_x = None
                second_f = float('inf')
                for s in starts:
                    if np.linalg.norm(s - best_x) > 1.0:
                        # Already evaluated
                        f = wrapped(s)
                        if f < second_f:
                            second_f = f
                            second_x = s.copy()

                if second_x is not None and self._call_count < budget - 10:
                    try:
                        result2 = minimize(
                            wrapped,
                            second_x,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': max(1, budget - self._call_count - 5),
                                     'ftol': 1e-12,
                                     'gtol': 1e-8,
                                     'maxls': 20}
                        )
                        if result2.fun < best_f:
                            best_f = result2.fun
                            best_x = result2.x
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
