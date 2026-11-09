"""Model-and-optimize approach for BBOB f20 (Schwefel) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Strategy:
    1. Quick multi-start L-BFGS-B from a few random points to get a good local solution fast.
    2. Refine with (1+λ)-ES using 1/5 success rule.
    3. Final high-precision L-BFGS-B polish.
    4. Explicit deadline guard to stay within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._safety_margin = 30  # seconds

    def __call__(self, func):
        self._start_time = time.time()
        self._deadline = self._start_time + 330 - self._safety_margin
        self._call_count = 0
        self.budget_remaining = self.budget

        def safe_call(x):
            """Wrapper to track calls and enforce budget/time limits."""
            nonlocal self
            if self._call_count >= self.budget:
                return 1e10
            if time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return val

        dim = self.dim
        rng = np.random.default_rng(42)

        # Track best solution found so far
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Quick multi-start with L-BFGS-B
        # Use 3 starts with tighter tolerances to save budget for ES
        num_starts = 3
        for i in range(num_starts):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break

            # Generate starting point
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = rng.uniform(-3, 3, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            # Clip to bounds
            x0 = np.clip(x0, -5, 5)

            # Try L-BFGS-B with tighter tolerances for faster convergence
            try:
                from scipy.optimize import minimize

                result = minimize(
                    safe_call,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6}
                )

                if result.fun < best_f:
                    update_best(result.x, result.fun)
            except Exception:
                pass

            # If we're running low on budget, skip remaining starts
            if self._call_count >= self.budget * 0.4:
                break

        # Ensure we have a best solution
        if best_x is None:
            x0 = rng.uniform(-5, 5, dim)
            f0 = safe_call(x0)
            best_x = x0.copy()
            best_f = f0

        # Phase 2: Refine with (1+λ)-ES using 1/5 success rule
        if self._call_count < self.budget and time.time() < self._deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 1.5  # Moderate initial step size
            lam = 4  # Population size

            # 1/5 success rule parameters
            success_count = 0
            total_count = 0
            window_size = 5

            while self._call_count < self.budget and time.time() < self._deadline:
                # Generate λ candidates
                candidates = []
                for _ in range(lam):
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = safe_call(y)
                    candidates.append((fy, y))

                # Select the best candidate
                candidates.sort(key=lambda t: t[0])
                best_fy, best_y = candidates[0]

                if best_fy < fx:
                    x = best_y
                    fx = best_fy
                    success_count += 1
                    update_best(x, fx)
                else:
                    pass

                total_count += 1

                # Adjust sigma based on 1/5 success rule
                if total_count >= window_size:
                    success_rate = success_count / total_count
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9
                    success_count = 0
                    total_count = 0

                # Clamp sigma
                sigma = np.clip(sigma, 0.005, 5.0)

                # If we're very close to the budget, reduce lambda to save calls
                if self._call_count > self.budget * 0.85:
                    lam = 1

        # Phase 3: Final high-precision L-BFGS-B polish
        if self._call_count < self.budget * 0.9 and time.time() < self._deadline:
            try:
                from scipy.optimize import minimize

                result = minimize(
                    safe_call,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
                )

                if result.fun < best_f:
                    update_best(result.x, result.fun)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
